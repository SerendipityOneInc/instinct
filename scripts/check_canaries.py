#!/usr/bin/env python3
"""Check an install against a model's published canaries.

python scripts/check_canaries.py --model <name> [--weights <dir|repo>]

Every model passes when all predictions agree and the largest absolute
difference of any candidate probability stays within ``canary_tolerance``
(model.json, probability units, the same unit for every model).

Record-level canaries (canary/records.jsonl) also report whether the raw logits
match the reference exactly (``exact_logit_match``) and the largest logit
difference. An exact match is expected only on the reference hardware and
software stack; other GPUs or kernels shift bf16 logits slightly, which the
probability tolerance absorbs. SystemOne canaries (canary/requests.jsonl) come
from a vLLM production deployment whose bf16 kernels differ from this HF path.
"""

import argparse
import json
import sys
from pathlib import Path

from instinct import InstinctModel, answer

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("id", "group_id", "state", "instructions", "primitive", "criteria")


def model_directory(name):
    """models/<dir> or reference/<dir> whose model.json declares ``name``."""
    for meta_path in sorted(ROOT.glob("*/*/model.json")):
        if meta_path.parent.parent.name in ("models", "reference"):
            if json.loads(meta_path.read_text())["name"] == name:
                return meta_path.parent
    raise SystemExit(f"no model directory declares name {name!r}")


def probabilities(item):
    if item["type"] == "noul":
        return {"yes": item["noul"], "no": 1 - item["noul"]}
    return item["probabilities"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    directory = model_directory(args.model)
    meta = json.loads((directory / "model.json").read_text())
    model = InstinctModel.from_pretrained(args.model, weights=args.weights)
    reference = [json.loads(line) for line in open(directory / "canary/reference.jsonl")]
    tolerance = meta["canary_tolerance"]
    worst, agree, total = 0.0, 0, 0
    summary = {"model": args.model}
    if (directory / "canary/records.jsonl").is_file():
        by_id = {row["id"]: row for row in reference}
        worst_logit, exact = 0.0, True
        for line in open(directory / "canary/records.jsonl"):
            record = json.loads(line)
            result = model.decide({k: record[k] for k in FIELDS})
            ref = by_id[record["id"]]
            exact &= result["logits"] == ref["logits"]
            worst_logit = max(worst_logit, max(abs(a - b) for a, b in zip(result["logits"], ref["logits"])))
            a, b = result["probabilities"], ref["probabilities"]
            worst = max(worst, max(abs(a[k] - b[k]) for k in b))
            agree += result["prediction"] == ref["prediction"]
            total += 1
        summary.update(exact_logit_match=exact, max_abs_logit_diff=worst_logit)
    else:
        for row in reference:
            mine = answer(model, row["request"])["answers"]
            for qid, item in row["answers"].items():
                a, b = probabilities(mine[qid]), probabilities(item)
                worst = max(worst, max(abs(a[k] - b[k]) for k in b))
                agree += max(a, key=a.get) == max(b, key=b.get)
                total += 1
    ok = agree == total and worst <= tolerance
    summary.update(items=total, same_prediction=agree, max_abs_probability_diff=worst,
                   tolerance=tolerance, ok=ok)
    print(json.dumps(summary))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
