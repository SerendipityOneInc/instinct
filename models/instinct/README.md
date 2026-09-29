# instinct

A decision model that is the released, untrained `Qwen/Qwen3.8-27B` (pinned revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`) behind our jqv prompt and readout. There are no weights of our own. It serves all three question types (`noul`, `choice`, `score`) from one forward pass per question, with a single option order.

- Prompt: jqv prompt format version 1, hash `4f85a0b34776`. The system message and document are the prefix. The question, lettered options and the assistant turn (thinking disabled) up to `Answer:` are the suffix.
- Readout: logits of the option-letter tokens (` A`, ` B`, ...) at the last position, softmax over the candidates at T = 1.0. The output is uncalibrated.
- Limit: 32768 prompt tokens, no truncation.
- Hardware: one H200, about 54 GB in bf16.
- Metadata: [`model.json`](model.json)

## Run

```bash
instinct-decide --model instinct models/instinct/examples/request.json
```

## Verify your install

`canary/requests.jsonl` holds 15 SystemOne requests with 18 questions. `canary/reference.jsonl` holds the answers our production deployment returned for them:

```bash
python scripts/check_canaries.py --model instinct
```

Production serves this model with vLLM and our patched readout; this repository runs a plain Transformers path. The prompts and label tokens are identical, and all 18 questions give the same answer. Probabilities can differ by up to 0.05 (`canary_tolerance` in `model.json`), because bf16 kernels differ between the two paths. On 2026-09-29 the largest difference was 0.032.

## Results

JevBench public 231, measured on one H200 with bf16 weights, one option order and no tuning on benchmark labels (batch `20260922T152934Z`):

| Split | Accuracy |
|---|---|
| easy | 48/48 |
| standard | 69/72 |
| hard | 84/111 |

Probabilities are uncalibrated (T = 1). These public items were also used during our development, so treat the table as a reference point, not a held-out score. The numbers come from our internal evaluation stack, not from this repository's Transformers path.

## Credits

Prompt format and readout: jqv, our internal decision-prompt library, published here with its maintainer's approval; the recipe in `instinct/recipes/jqv_27b.py` is a port of it. Base model: Qwen3.8-27B by the Qwen team (see its model card for license).
