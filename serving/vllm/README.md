# Optional vLLM serving for the reference baseline

This directory is **optional**. The `instinct/` package (plain Transformers,
`instinct-serve`) is the **reference implementation**: prompts, label tokens,
option orders, temperature and merging are defined there, and this directory
imports them from `instinct.recipes`. Nothing here changes what the models
compute; it only replaces the forward pass with a patched vLLM server. If the
two ever disagree, the Transformers path is correct. It is used for the
`reference-qwen3.8-27b` baseline (`reference/qwen3.8-27b/`).

The service exposes the same API as `instinct-serve`: `POST /v1/systemone`
and `GET /health`.

## Pinned versions

- vLLM **0.17.1** exactly (`vllm.__version__`). The patch was written and hash
  checked against the v0.17.1 sources only; it has not been tried on any other
  version. `common/build_overlay.py` refuses files whose SHA-256 differs from
  the 0.17.1 originals.
- Hardware used by the original deployment: one NVIDIA H200 (BF16, TP1,
  32768 context). Other GPUs were not measured.

## What the optimizations do

### Shared: native readout patch (`patch/native-readout.patch`)

Five vLLM files are modified (`patch/patch-manifest.json` lists their
before/after hashes). Stock vLLM rewrites an `echo` / `max_tokens=0` scoring
request into one generated token and runs the sampler. The patch:

1. `protocol.py`: keeps `max_tokens=0` instead of rewriting it to 1.
2. `sampling_params.py`: accepts `max_tokens=0` when `prompt_logprobs` is set.
3. `gpu_model_runner.py`: when `max_tokens == 0`, skips the next-token sampler
   entirely; for requests with `vllm_xargs.jqv_readout=native-v1`, computes the
   LM head on the **single final prompt position**, applies `log_softmax` in
   float32 over the full vocabulary and gathers only the requested candidate
   token ids (2 to 26), instead of the per-position prompt-logprob path.
4. `scheduler.py`: finishes such a request as soon as prefill completes, with
   zero generated tokens, and appends a per-sequence line to the file named by
   the required environment variable `JQV_PREFILL_AUDIT` (use `/dev/null` to
   discard).
5. `serving.py`: returns the values as `jqv_candidate_logprobs` and
   `jqv_readout_position` on each choice, with empty text and token ids.

The patch also contains a `jqv_compact_legacy` response mode (compact response
for the older candidate-appended scoring). The adapter here does not use it.

### `reference-qwen3.8-27b` (frozen Qwen3.8-27B, jqv prompt hash `4f85a0b34776`)

One independent sequence per question is sent in a single `/v1/completions`
request with `echo=true`, `max_tokens=0`, `prompt_logprobs=len(ids)-1` and the
native-v1 readout, so the server does one prefill per question and reads the
candidate letter (` A`, ` B`, ...) log-probabilities at the last position with
no sampling. The vLLM server runs with compilation mode 3, `PIECEWISE` CUDA
graphs (capture sizes 128 to 2048), prefix caching off, async scheduling off
(`reference-27b/serve.sh:24-30`; request construction in
`common/vllm_readout.py:76-92`). These are the settings of the original
deployment.

## Files

| Path | Purpose |
|---|---|
| `LICENSE`, `NOTICE` | Apache-2.0 text and attribution for the vLLM-derived files |
| `patch/native-readout.patch`, `patch/patch-manifest.json` | the patch and its hashes |
| `common/build_overlay.py` | builds the five patched files from a vLLM 0.17.1 install |
| `common/check_overlay.py` | verifies the installed vLLM carries the overlay files |
| `common/vllm_readout.py` | vLLM transport + `POST /v1/systemone` server using `instinct` recipes |
| `common/serve_adapter.py` | CLI for that server |
| `reference-27b/serve.sh` | launches vLLM and the adapter for `reference-qwen3.8-27b` |

## Apply and launch

Prerequisites: `pip install -e .` from the repository root (provides
`instinct`), `pip install vllm==0.17.1`, `transformers` for the tokenizer, a
local model directory, and the `patch` utility.

```bash
cd serving/vllm
SITE=$(python3 -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))")
python3 common/build_overlay.py --vllm-root "$SITE" --output overlay-native
```

The overlay is a tree of five files. Install it by copying them over the same
paths in the vLLM package (`cp -r overlay-native/vllm/. "$SITE/vllm/"`), or bind
mount each file read-only over the installed one in a container. Back up the
originals first; the patch does not modify anything in place.

### reference-qwen3.8-27b

```bash
python3 common/check_overlay.py --manifest overlay-native/patch-manifest.json
MODEL_DIR=/path/to/Qwen3.8-27B \
OVERLAY_MANIFEST=$PWD/overlay-native/patch-manifest.json \
  reference-27b/serve.sh     # adapter on 127.0.0.1:8008
```

## Cautions

- The adapter has no authentication or TLS. It and vLLM bind
  to 127.0.0.1 by default; put your own gateway in front before exposing them.
- The patch asserts unsupported cases (mixed scoring and generation batches,
  async scheduling, speculative decoding). Do not enable those.
- The vLLM server accepts only its native readout requests for this model;
  do not use the patched server for ordinary chat/completions.
- No latency or accuracy numbers are given here. Verify parity yourself with
  `scripts/parity_check.py` and `scripts/compare_endpoint.py` against the
  reference server before relying on the service.
- The adapter uses the union of candidate label ids per request; the original
  27B adapter always sent all 26 letters. The values read are log-softmax over
  the full vocabulary either way, but this exact path has not been re-run
  end to end after porting.
