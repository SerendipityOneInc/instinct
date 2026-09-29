# instinct-tuned-4b

A 4B decision model: a LoRA fine-tune of Qwen3.5-4B, merged into the base weights. It serves all three question types (`noul`, `choice`, `score`) from one forward pass at full depth, with serving temperature T = 2.80.

- Weights and model card: [`srpone/instinct-tuned-4b`](https://huggingface.co/srpone/instinct-tuned-4b)
- Metadata: [`model.json`](model.json)

## Run

```bash
instinct-decide --model srpone/instinct-tuned-4b models/instinct-tuned-4b/examples/request.json
```

[`examples/expected.json`](examples/expected.json) holds the answer from the reference hardware.

## Verify your install

`canary/` holds 14 synthetic requests and the logits our production deployment returned for them:

```bash
python scripts/check_canaries.py --model instinct-tuned-4b
```

On 1x H200 with bf16 and `transformers==5.16.1`, the logits match exactly (`canary_tolerance` 0). On other hardware, expect small differences.

## Results

JevBench public 231 at full depth, bf16, one option order. See the model card for training details and limitations.

| Split | Accuracy |
|---|---|
| easy | 48/48 |
| standard | 69/72 |
| hard | 77/111 |
