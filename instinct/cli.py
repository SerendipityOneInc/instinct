"""Answer a SystemOne request file: instinct-decide --model <name> request.json"""

import argparse
import json
import sys

from .model import InstinctModel
from .systemone import answer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="SystemOne request JSON file, or - for stdin")
    parser.add_argument("--model", required=True, help="released model name, e.g. instinct-tuned-4b")
    parser.add_argument("--weights", default=None,
                        help="local directory or HF repo id overriding the pinned weights")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args(argv)
    source = sys.stdin if args.request == "-" else open(args.request, encoding="utf-8")
    with source:
        body = json.load(source)
    model = InstinctModel.from_pretrained(
        args.model, weights=args.weights, revision=args.revision, device=args.device,
        dtype=args.dtype,
    )
    json.dump(answer(model, body), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
