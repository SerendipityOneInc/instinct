# ZooWork Instinct Tuned 4B

`instinct-tuned-4b` is ZooWork's open 4B decision model: a LoRA fine-tune of Qwen3.5-4B, merged into the base weights. It serves all three question types (`noul`, `choice`, `score`) from one forward pass at full depth, with serving temperature T = 2.80 for `choice` and `score` and T = 0.50 for `noul`. JevBench v1.5 scores a `noul` answer with P(yes) strictly between 0.2 and 0.8 as an abstention, so `noul` probabilities are sharpened with a separate temperature chosen on an in-house development set; predictions are unchanged, since a two-class softmax at a different temperature keeps the arg-max.

- Weights and model card: [`srpone/instinct-tuned-4b`](https://huggingface.co/srpone/instinct-tuned-4b)
- Hosted API: [`instinct.zoowork.ai`](https://instinct.zoowork.ai/) using model ID `instinct-tuned-4b`
- Reference runtime: [`SerendipityOneInc/instinct`](https://github.com/SerendipityOneInc/instinct)
- Metadata: [`model.json`](model.json)

## Run

```bash
instinct-decide --model instinct-tuned-4b models/instinct-tuned-4b/examples/request.json
```

[`examples/expected.json`](examples/expected.json) holds the answer from the reference hardware.

## Verify your install

`canary/` holds 14 synthetic requests and the logits our production deployment returned for them:

```bash
python scripts/check_canaries.py --model instinct-tuned-4b
```

The check passes when every prediction agrees and every candidate probability is within `canary_tolerance` (0.05, in probability units after the temperature softmax, T = 2.80, or 0.50 for the `noul` canary; see `model.json`). It also reports `exact_logit_match` separately: on 1x H200 with bf16 and `transformers==5.16.1` the logits matched the reference exactly. Other GPUs and kernels shift bf16 logits slightly, so an inexact logit match alone is not a failure: on an A10 in the NGC `pytorch:25.06-py3` container, all 14 predictions agree, the largest logit difference is 0.25 and the largest probability difference is 0.020.

## Measured results

JevBench public 231 (198/231) at full depth, bf16, scored with the official JevBench client, items in their original option order. These public items were also used during our development for model selection, so treat the table as a reference point, not a held-out score. See the model card for training details and limitations.

| Scope | Correct | Accuracy |
|---|---:|---:|
| All published tasks | **198/231** | **85.71%** |
| easy | 48/48 | 100.00% |
| standard | 69/72 | 95.83% |
| hard | 81/111 | 72.97% |

| Serving latency | Value |
|---|---:|
| Direct-upstream p50 | 62 ms |
| Direct-upstream p95 | ≈110 ms (estimated) |

Latency refers to warmed, serial serving and excludes public Internet, TLS and gateway overhead. The p50 is measured; the p95 is an estimate from the measured direct-upstream p50 and observed end-to-end tail spread, not an SLO.

---

Part of the Instinct model family by [ZooWork](https://zoowork.ai) · [instinct.zoowork.ai](https://instinct.zoowork.ai/)
