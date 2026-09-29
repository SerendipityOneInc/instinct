"""SystemOne HTTP adapter in front of one or two native-readout vLLM servers.

    python serve_adapter.py --model instinct --weights <model dir> \
        --backend http://127.0.0.1:8001 --backend-model instinct-backend --readout native

Serves POST /v1/systemone and GET /health, like ``instinct-serve``.
Set VLLM_API_KEY if the vLLM servers were started with --api-key.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "instinct-dual-4b"))  # parallel_order_scores

from vllm_readout import Backend, VLLMDecisionModel, run_server  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="instinct recipe name, e.g. instinct")
    p.add_argument("--weights", required=True, help="local model directory (tokenizer + weights)")
    p.add_argument("--backend", required=True, help="primary vLLM base URL")
    p.add_argument("--backend-secondary", help="second vLLM base URL (dual-order mode)")
    p.add_argument("--backend-model", required=True, help="--served-model-name of the vLLM servers")
    p.add_argument("--served-name", help="model id exposed by this adapter (default: --model)")
    p.add_argument("--readout", choices=("native", "native-padded"), required=True)
    p.add_argument("--backend-batch-size", type=int, default=0,
                   help="sequences per vLLM request; 0 = whole request. Dual mode requires 1")
    p.add_argument("--backend-timeout", type=float, default=1200.0)
    p.add_argument("--model-revision", default=None, help="informational, echoed by /health")
    p.add_argument("--reflex-revision", default=None, help="informational, echoed by /health")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8008)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    from instinct.recipes import get as get_recipe

    recipe = get_recipe(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.weights, local_files_only=True)
    key = os.environ.get("VLLM_API_KEY") or None

    def backend(url):
        return Backend(url, args.backend_model, args.backend_timeout, key)

    model = VLLMDecisionModel(
        tokenizer,
        recipe,
        backend(args.backend),
        backend(args.backend_secondary) if args.backend_secondary else None,
        padded=args.readout == "native-padded",
        batch_size=args.backend_batch_size,
    )
    identity = {
        "readout": args.readout,
        "backend_version": "0.17.1-native-readout-patch",
        "temperature": recipe.temperature,
        "dual_backend": bool(args.backend_secondary),
        "model_revision": args.model_revision or recipe.revision,
        "reflex_revision": args.reflex_revision,
    }
    run_server(model, args.served_name or args.model, identity, args.host, args.port)


if __name__ == "__main__":
    main()
