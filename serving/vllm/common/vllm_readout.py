"""Score instinct recipes through a native-readout-patched vLLM 0.17.1 server.

The prompt, label tokens, option orders, temperature and merge all come from the
``instinct`` recipe (the reference implementation); only the forward pass is
delegated to vLLM. For each branch the patched server returns the log-softmax
values of the candidate label tokens at ONE position, with zero generated
tokens (see ../README.md). ``recipe.merge`` then applies the temperature
softmax, maps back to candidate ids and averages branches exactly as the
Transformers path does.

Requires the ``instinct`` package to be importable (``pip install -e .`` in the
repository root).
"""

import http.client
import json
import logging
import math
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from instinct._vendor.apus_runtime.contracts import format_response
from instinct.systemone import to_answer

MAX_CANDIDATES = 26  # the patch accepts 2..26 candidate ids per request


class Backend:
    """Minimal persistent-connection JSON client for one vLLM server."""

    def __init__(self, url, model, timeout=1200.0, api_key=None):
        parts = urllib.parse.urlsplit(url)
        self.host, self.port = parts.hostname, parts.port or 80
        self.model, self.timeout, self.api_key = model, timeout, api_key
        self._conn = None
        self._lock = threading.Lock()

    def completions(self, payload):
        body = json.dumps({"model": self.model, **payload}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with self._lock:
            for attempt in (0, 1):
                try:
                    if self._conn is None:
                        self._conn = http.client.HTTPConnection(
                            self.host, self.port, timeout=self.timeout
                        )
                    self._conn.request("POST", "/v1/completions", body, headers)
                    response = self._conn.getresponse()
                    raw = response.read()
                    break
                except (OSError, http.client.HTTPException):
                    self._conn = None  # stale keep-alive connection; scoring is idempotent
                    if attempt:
                        raise
        if response.status != 200:
            raise RuntimeError(f"vLLM returned {response.status}: {raw[:300]!r}")
        return json.loads(raw)

    def healthy(self):
        try:
            with urllib.request.urlopen(
                f"http://{self.host}:{self.port}/health", timeout=2
            ):
                return True
        except Exception:
            return False


def score_chunk(backend, prompts, label_ids, all_ids, padded):
    """One request of independent sequences -> label log-probabilities per sequence.

    ``padded``: the prompts already carry one extra ignored token and the server
    reads the penultimate position (readout ``native-padded-v1``).
    """
    if not 2 <= len(all_ids) <= MAX_CANDIDATES:
        raise RuntimeError(f"native candidate union has invalid size: {len(all_ids)}")
    response = backend.completions(
        {
            "prompt": prompts,
            "echo": True,
            "max_tokens": 0,
            "prompt_logprobs": len(all_ids) - 1,
            "temperature": 0.0,
            "vllm_xargs": {
                "jqv_readout": "native-padded-v1" if padded else "native-v1",
                "jqv_candidate_ids": ",".join(map(str, all_ids)),
            },
        }
    )
    usage = response.get("usage")
    if not isinstance(usage, dict) or usage.get("prompt_tokens") != sum(map(len, prompts)):
        raise RuntimeError("backend prompt-token mismatch")
    if usage.get("completion_tokens") != 0:
        raise RuntimeError("patched backend performed token generation")
    choices = response["choices"]
    if len(choices) != len(prompts):
        raise RuntimeError("backend returned the wrong number of choices")
    ordered = [None] * len(prompts)
    for choice in choices:
        i = choice["index"]
        if not (type(i) is int and 0 <= i < len(prompts)) or ordered[i] is not None:
            raise RuntimeError("invalid choice index")
        ordered[i] = choice
    scores = []
    for choice, expected_ids, prompt in zip(ordered, label_ids, prompts):
        if choice["text"] != "" or choice["token_ids"] != []:
            raise RuntimeError("readout choice carries generated content")
        if choice.get("jqv_readout_position") != len(prompt) - (2 if padded else 1):
            raise RuntimeError("native readout position mismatch")
        union = {int(k): v for k, v in choice["jqv_candidate_logprobs"].items()}
        if set(union) != set(all_ids) or not all(
            isinstance(v, (int, float)) and math.isfinite(v) and v <= 1e-5
            for v in union.values()
        ):
            raise RuntimeError("invalid candidate log-probabilities")
        scores.append([float(union[t]) for t in expected_ids])
    return scores


class VLLMDecisionModel:
    """Drop-in for ``instinct.model.InstinctModel`` on the server side.

    ``secondary`` (optional): a second, independent vLLM server on another GPU.
    With it, each question's two option-order branches run concurrently, one per
    server, and requests are serialized (batch one) so the two engines never
    batch unrelated sequences.
    """

    def __init__(self, tokenizer, recipe, primary, secondary=None, padded=False, batch_size=0):
        self.tokenizer, self.recipe = tokenizer, recipe
        self.primary, self.secondary, self.padded = primary, secondary, padded
        self.batch_size = batch_size  # 0 = all sequences of a request in one call
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(2, "order") if secondary else None
        if secondary and (batch_size != 1 or not padded):
            raise ValueError("dual backends require batch size 1 and the padded readout")

    def _sequences(self, branches):
        if not self.padded:
            return [list(e.input_ids) for e in branches]
        return [list(e.input_ids) + [e.label_token_ids[0]] for e in branches]

    def _score(self, backend, encoded):
        all_ids = sorted({t for e in encoded for t in e.label_token_ids})
        prompts = self._sequences(encoded)
        labels = [e.label_token_ids for e in encoded]
        step = self.batch_size or len(prompts)
        out = []
        for s in range(0, len(prompts), step):
            out += score_chunk(
                backend, prompts[s : s + step], labels[s : s + step], all_ids, self.padded
            )
        return out

    def _score_pair(self, encoded):
        from parallel_order_scores import score_order_pairs  # instinct-dual-4b/

        all_ids = sorted({t for e in encoded for t in e.label_token_ids})
        return score_order_pairs(
            self.executor,
            score_chunk,
            self.primary,
            self.secondary,
            self._sequences(encoded),
            [e.label_token_ids for e in encoded],
            all_ids,
            self.padded,
        )

    def decide_many(self, records):
        """Score every record of one request; batch across questions when single-engine."""
        per_record = [
            [self.recipe.encode(self.tokenizer, r, o) for o in self.recipe.orders(r)]
            for r in records
        ]
        with self.lock:
            if self.secondary is not None:
                logits = [self._score_pair(enc) for enc in per_record]
            else:
                flat = [e for enc in per_record for e in enc]
                scored, k, logits = self._score(self.primary, flat), 0, []
                for enc in per_record:
                    logits.append(scored[k : k + len(enc)])
                    k += len(enc)
        results = []
        for record, enc, lg in zip(records, per_record, logits):
            probabilities = self.recipe.merge(record, enc, lg)
            response = format_response(record, probabilities)
            response.update(
                prediction=max(response["probabilities"], key=response["probabilities"].get),
                prompt_tokens=sum(len(e.input_ids) for e in enc),
                temperature=self.recipe.temperature,
            )
            results.append(response)
        return results

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


def answer_many(model, body, model_id=None):
    """Same request/response contract as instinct.systemone.answer."""
    records = model.recipe.records(body, model_id)
    results = model.decide_many([r for _, _, r in records])
    answers = {qid: to_answer(kind, res) for (qid, kind, _), res in zip(records, results)}
    return {
        "answers": answers,
        "usage": {
            "input_tokens": sum(r["prompt_tokens"] for r in results),
            "output_tokens": 0,
        },
    }


def make_handler(model, model_id, identity):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # no per-request access log on the hot path
            pass

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": "not found"})
            backends = [b for b in (model.primary, model.secondary) if b is not None]
            if not all(b.healthy() for b in backends):
                return self._send(503, {"error": "vLLM backend not ready"})
            self._send(200, {"status": "ok", "model": model_id, **identity})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                started = time.perf_counter()
                result = answer_many(model, body, model_id)
            except (ValueError, TypeError) as exc:
                return self._send(422, {"error": str(exc)})
            except Exception as exc:  # backend failure
                logging.exception("vLLM scoring failed")
                return self._send(502, {"error": str(exc)})
            result.update(model=model_id, runtime={"server_s": time.perf_counter() - started})
            self._send(200, result)

    return Handler


def run_server(model, model_id, identity, host, port):
    server = ThreadingHTTPServer((host, port), make_handler(model, model_id, identity))
    print(f"serving {model_id} on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        model.close()
