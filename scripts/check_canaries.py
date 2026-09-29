#!/usr/bin/env python3
"""Check an install against a model's published canaries.

python scripts/check_canaries.py --model <name> [--weights <dir|repo>]

Record-level canaries (canary/records.jsonl) are compared logit by logit;
SystemOne canaries (canary/requests.jsonl) are compared answer by answer within
the model's canary_tolerance (model.json), because their reference comes from a
vLLM production deployment whose bf16 kernels differ from this HF path.
"""

import argparse
import json
import sys
from pathlib import Path

from instinct import InstinctModel, answer

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("id", "group_id", "state", "instructions", "primitive", "criteria")


def probabilities(item):
    if item["type"] == "noul":
        return {"yes": item["noul"], "no": 1 - item["noul"]}
    return item["probabilities"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    directory = ROOT / "models" / args.model
    meta = json.loads((directory / "model.json").read_text())
    model = InstinctModel.from_pretrained(args.model, weights=args.weights)
    reference = [json.loads(line) for line in open(directory / "canary/reference.jsonl")]
    worst, agree, total = 0.0, 0, 0
    if (directory / "canary/records.jsonl").is_file():
        tolerance = meta.get("canary_tolerance", 0.0)
        by_id = {row["id"]: row for row in reference}
        for line in open(directory / "canary/records.jsonl"):
            record = json.loads(line)
            result = model.decide({k: record[k] for k in FIELDS})
            ref = by_id[record["id"]]
            worst = max(worst, max(abs(a - b) for a, b in zip(result["logits"], ref["logits"])))
            agree += result["prediction"] == ref["prediction"]
            total += 1
        unit = "logit"
    else:
        tolerance = meta["canary_tolerance"]
        for row in reference:
            mine = answer(model, row["request"])["answers"]
            for qid, item in row["answers"].items():
                a, b = probabilities(mine[qid]), probabilities(item)
                worst = max(worst, max(abs(a[k] - b[k]) for k in b))
                agree += max(a, key=a.get) == max(b, key=b.get)
                total += 1
        unit = "probability"
    ok = agree == total and worst <= tolerance
    print(json.dumps({"model": args.model, "items": total, "same_prediction": agree,
                      f"max_abs_{unit}_diff": worst, "tolerance": tolerance, "ok": ok}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
