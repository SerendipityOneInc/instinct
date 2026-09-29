"""Minimal single-GPU HTTP server: POST /v1/systemone, GET /health.

instinct-serve --model <name> [--weights <dir|repo>] [--host 127.0.0.1] [--port 8008]

Requests are served one at a time. This is a reference server, not a
production deployment (no batching, auth, or admission control).
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .model import InstinctModel
from .systemone import answer


def make_handler(model, model_id):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": model_id})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                started = time.perf_counter()
                with lock:
                    result = answer(model, body, model_id)
            except (ValueError, TypeError) as exc:
                return self._send(422, {"error": str(exc)})
            except Exception as exc:  # e.g. CUDA out of memory
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            result.update(model=model_id, runtime={"server_s": time.perf_counter() - started})
            self._send(200, result)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args(argv)
    model_id = args.model
    model = InstinctModel.from_pretrained(args.model, weights=args.weights,
                                          revision=args.revision, device=args.device,
                                          dtype=args.dtype)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, model_id))
    print(f"serving {model_id} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
