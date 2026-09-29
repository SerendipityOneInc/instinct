#!/usr/bin/env python3
"""Compare InstinctModel answers with a running /v1/systemone endpoint.

Each line of --requests is a SystemOne body. The same body (with "model" set to
--endpoint-model) is sent to --url, and per-question probabilities are compared.
The endpoint's answers (and ours, as local_answers) are written to --output.
"""

import argparse
import json
import sys
import urllib.request

from instinct import InstinctModel, answer


def probabilities(item):
    if item["type"] == "noul":
        return {"yes": item["noul"]}
    return item["probabilities"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="released model name")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--url", required=True, help="base URL of the endpoint")
    parser.add_argument("--endpoint-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=0.02)
    args = parser.parse_args()
    model = InstinctModel.from_pretrained(args.model, weights=args.weights)
    worst, same, total = 0.0, 0, 0
    with open(args.output, "x") as out:
        for line in open(args.requests, encoding="utf-8"):
            body = json.loads(line)
            mine = answer(model, body)["answers"]
            request = dict(body, model=args.endpoint_model)
            with urllib.request.urlopen(urllib.request.Request(
                    args.url.rstrip("/") + "/v1/systemone", json.dumps(request).encode(),
                    {"Content-Type": "application/json"}), timeout=300) as response:
                theirs = json.load(response)["answers"]
            out.write(json.dumps({"request": body, "answers": theirs, "local_answers": mine},
                                 ensure_ascii=False) + "\n")
            for qid, item in theirs.items():
                a, b = probabilities(mine[qid]), probabilities(item)
                diff = max(abs(a[k] - b[k]) for k in b)
                worst = max(worst, diff)
                total += 1
                same += max(a, key=a.get) == max(b, key=b.get)
                print(json.dumps({"question": qid, "max_abs_prob_diff": round(diff, 6),
                                  "same_argmax": max(a, key=a.get) == max(b, key=b.get)}))
    summary = {"questions": total, "same_argmax": same, "max_abs_prob_diff": worst}
    print(json.dumps(summary))
    sys.exit(0 if same == total and worst <= args.atol else 1)


if __name__ == "__main__":
    main()
