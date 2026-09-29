# instinct-tuned-4b

A 4B decision model: a LoRA fine-tune of Qwen3.5-4B, merged into the base weights. It serves all three question types (`noul`, `choice`, `score`) from one forward pass at full depth, with serving temperature T = 2.80.

- Weights and model card: [`srpone/instinct-tuned-4b`](https://huggingface.co/srpone/instinct-tuned-4b)
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

On the reference environment (1x H200, bf16, torch 2.13, `transformers==5.16.1`) the logits match production exactly. On other GPUs, bf16 kernels differ slightly: on an A10 in the NGC `pytorch:25.06-py3` container all 14 predictions match and the largest logit difference is 0.25 (about two bf16 steps at these magnitudes). `canary_tolerance` in `model.json` is 0.5 logits.

## Results

JevBench public 231 at full depth, bf16, scored with the official JevBench client with items in their original option order (198/231 overall). These public items were also used during our development for model selection, so treat the table as a reference point, not a held-out score. See the model card for training details and limitations.

| Split | Accuracy |
|---|---|
| easy | 48/48 |
| standard | 69/72 |
| hard | 81/111 |

---

Part of the Instinct model family by [ZooWork](https://zoowork.ai) · [instinct.zoowork.ai](https://instinct.zoowork.ai/)
