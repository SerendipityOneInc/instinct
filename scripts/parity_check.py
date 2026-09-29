#!/usr/bin/env python3
"""Compare InstinctModel logits with a reference run, item by item.

records.jsonl: one request record per line (id, group_id, state, instructions,
primitive, criteria). reference.jsonl: {"id", "logits", "prediction", ...} per
line; for multi-order models, "logits" is the concatenation of the
branches in order. Exits nonzero unless every item matches bitwise (or within --atol).
"""

import argparse
import json
import sys

from instinct import InstinctModel

FIELDS = ("id", "group_id", "state", "instructions", "primitive", "criteria")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="released model name")
    parser.add_argument("--weights", default=None, help="local directory or HF repo id")
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    reference = {}
    for line in open(args.reference):
        row = json.loads(line)
        reference[row["id"]] = row
    model = InstinctModel.from_pretrained(args.model, weights=args.weights)
    compared = exact = argmax = 0
    worst = 0.0
    with open(args.output, "x") as out:
        for path in args.records:
            for line in open(path):
                record = json.loads(line)
                if record["id"] not in reference:
                    continue
                result = model.decide({k: record[k] for k in FIELDS})
                ref = reference[record["id"]]
                mine = result["logits"] if "logits" in result else [
                    x for branch in result["branches"] for x in branch["logits"]]
                theirs = ref["logits"]
                if len(mine) != len(theirs):
                    raise ValueError(f"{record['id']}: logit count differs from the reference")
                diff = max(abs(a - b) for a, b in zip(mine, theirs))
                compared += 1
                exact += mine == theirs
                argmax += result["prediction"] == ref["prediction"]
                worst = max(worst, diff)
                out.write(json.dumps({"id": record["id"], "logits": mine,
                                      "prediction": result["prediction"],
                                      "max_abs_diff": diff}) + "\n")
    summary = {"compared": compared, "reference": len(reference), "bitwise_equal": exact,
               "same_prediction": argmax, "max_abs_logit_diff": worst}
    print(json.dumps(summary))
    ok = compared == len(reference) and worst <= args.atol
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
