#!/usr/bin/env bash
# Serve `instinct` (frozen Qwen3.8-27B, jqv prompt) on ONE GPU through the
# native-readout-patched vLLM 0.17.1, behind POST /v1/systemone.
#
#   MODEL_DIR=/path/to/Qwen3.8-27B ./serve.sh
#
# Optional: PORT (8008), BACKEND_PORT (8001), SERVED_BACKEND_NAME, JQV_PREFILL_AUDIT
# (per-sequence scheduler audit file; default /dev/null. Keep it on node-local
# disk if you enable it: the original deployment measured a large slowdown when
# the audit file lived on network storage.)
set -euo pipefail
: "${MODEL_DIR:?set MODEL_DIR to the local Qwen3.8-27B directory}"
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8008}
BACKEND_PORT=${BACKEND_PORT:-8001}
NAME=${SERVED_BACKEND_NAME:-instinct-backend}

# The patched scheduler requires this variable (see patch/native-readout.patch).
export JQV_PREFILL_AUDIT=${JQV_PREFILL_AUDIT:-/dev/null}
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false

python3 "$HERE/../common/check_overlay.py" --manifest "${OVERLAY_MANIFEST:?set OVERLAY_MANIFEST to <overlay>/patch-manifest.json}"

COMPILATION='{"mode": 3, "cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [128, 256, 512, 1024, 2048], "max_cudagraph_capture_size": 2048}'
vllm serve "$MODEL_DIR" --served-model-name "$NAME" \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --gpu-memory-utilization 0.75 --max-num-seqs 32 \
  --max-logprobs 32 --no-async-scheduling --no-enable-prefix-caching \
  --compilation-config "$COMPILATION" --cudagraph-metrics \
  --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null || true' EXIT INT TERM

until curl -sf "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null; do
  kill -0 "$BACKEND_PID" 2>/dev/null || { echo "vLLM exited before becoming ready" >&2; exit 1; }
  sleep 2
done

python3 "$HERE/../common/serve_adapter.py" --model instinct --weights "$MODEL_DIR" \
  --backend "http://127.0.0.1:$BACKEND_PORT" --backend-model "$NAME" \
  --readout native --host "${HOST:-127.0.0.1}" --port "$PORT"
