# instinct-dual-4b

A 4B decision model with no weights of our own: the frozen, untrained [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) driven by the [Reflex](https://github.com/kshetrajna12/reflex) prompt construction. It serves all three question types (`noul`, `choice`, `score`), scores every question under two option orders, and reports the average of the two softmaxes at temperature 1 (uncalibrated).

- Metadata: [`model.json`](model.json)
- Prompt: Reflex `prompt_style="markdown"` at revision `b77f03bab8e00d7319e64550e062c1eac8111875` (MIT, Copyright (c) 2026 Kshetrajna Raghavan; vendored subset in `instinct/_vendor/reflex`). The system message, the `# Evidence` / `# Criterion` / `# Options` layout, the lettered options, the empty think prefix and the option-order rule are Reflex's.
- Our contribution is the serving and inference optimizations around that prompt: the two option-order branches run concurrently on two workers, and the answer is read from a native padded readout. The HF path here reads the last position of the unpadded prompt, which is the same computation.

## How a question is scored

- The shared state and each question branch are tokenized separately and concatenated, so the state prefix is identical across branches.
- Options are lettered `A`, `B`, ... and the candidate tokens are the bare letters. `noul` is shown as `A. yes: ...` / `B. no: ...` (your `true` / `false` wording when supplied).
- Two orders per question: the order you gave, then one seeded shuffle (`random.Random("0:<question id>")`; the swapped order for two-option questions and `noul`). Each order gives a softmax over its letters at T = 1; probabilities are mapped back to your option ids and averaged.
- Input is never truncated: a question branch over 4096 tokens or a prompt over the 262144-token context is rejected.

## Run

```bash
instinct-decide --model instinct-dual-4b models/instinct-dual-4b/examples/request.json
```

Weights are downloaded from `Qwen/Qwen3.5-4B` at the pinned revision. Reference hardware: 1x H200, bf16, `transformers==5.16.1`.

## Verify your install

`canary/requests.jsonl` holds 15 SystemOne requests with 18 questions. `canary/reference.jsonl` holds the answers our production deployment returned for them:

```bash
python scripts/check_canaries.py --model instinct-dual-4b
```

Production serves this model with vLLM and our patched readout; this repository runs a plain Transformers path. The prompts and label tokens are identical, and all 18 questions give the same answer. Probabilities can differ by up to 0.02 (`canary_tolerance` in `model.json`), because bf16 kernels differ between the two paths. On 2026-09-29 the largest difference was 0.012.

## Results

JevBench public 231, measured on the production serving path (vLLM, bf16, two option orders, one H200 per order):

| Split | Accuracy |
|---|---|
| easy | 48/48 |
| standard | 68/72 |
| hard | 74/111 |

Probabilities are uncalibrated (T = 1). On the hard split, ECE is 0.089. With the two orders evaluated concurrently on two H200s, localhost latency is about 41 ms at p50 and 79–82 ms at p95. These public items were also used during our development, so treat the table as a reference point, not a held-out score.

## Credits

Prompt construction and option-order logic: Reflex by Kshetrajna Raghavan (MIT). Base model: Qwen3.5-4B by the Qwen team (see its model card for license).

---

Part of the Instinct model family by [ZooWork](https://zoowork.ai) · [instinct.zoowork.ai](https://instinct.zoowork.ai/)
