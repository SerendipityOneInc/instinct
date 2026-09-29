# Instinct

**[ZooWork](https://zoowork.ai)** · [Instinct homepage](https://instinct.zoowork.ai/)

Instinct is a **decision model** built by [ZooWork](https://zoowork.ai): `instinct-tuned-4b`, a post-trained Qwen3.5-4B. Given a shared state and a question with fixed candidates, it returns a probability per candidate from one forward pass, with no text generation.

This repository holds the runtime, the `instinct` Python package. A model is a *recipe* (`instinct/recipes/`): its weights, prompt, label tokens, readout, option orders and temperature.

## Model

| Model | Directory | Weights | Orders | T | Question types |
|---|---|---|---|---|---|
| instinct-tuned-4b | [`models/instinct-tuned-4b`](models/instinct-tuned-4b) | [`srpone/instinct-tuned-4b`](https://huggingface.co/srpone/instinct-tuned-4b) (LoRA fine-tune of Qwen3.5-4B, merged) | 1 | 2.80 (calibrated) | noul, choice, score |

The model directory contains:

- a README with a results table;
- `model.json` metadata;
- an example request with its expected answer;
- canary records with the logits our production deployment returned, and a tolerance (`scripts/check_canaries.py --model <name>`).

To add a model, see [`models/TEMPLATE`](models/TEMPLATE).

## Reference baseline

[`reference/qwen3.8-27b`](reference/qwen3.8-27b) (recipe `reference-qwen3.8-27b`) runs the untrained [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at a pinned revision behind our decision prompt and readout. It is provided as a reference baseline for comparison, not as a released Instinct model, and has no Hugging Face repository of its own: the weights are downloaded from Qwen.

## Install

A CUDA GPU is required. The runtime is verified on one H200 in bf16.

```bash
git clone https://github.com/SerendipityOneInc/instinct && cd instinct
pip install -e ".[gpu]"
```

Layer execution is verified against `transformers==5.16.1`, and the runtime refuses to start with any other version unless `INSTINCT_ALLOW_UNVERIFIED_TRANSFORMERS=1` is set. `flash-linear-attention` provides the Qwen3.5 linear-attention kernels.

## Use

Every command takes the recipe name: `instinct-tuned-4b` or `reference-qwen3.8-27b`.

```bash
instinct-decide --model instinct-tuned-4b models/instinct-tuned-4b/examples/request.json
instinct-decide --model reference-qwen3.8-27b reference/qwen3.8-27b/examples/request.json
instinct-serve  --model instinct-tuned-4b --port 8008
# --weights <local dir or HF repo> overrides where the weights come from
curl -s localhost:8008/v1/systemone -d @models/instinct-tuned-4b/examples/request.json
```

```python
from instinct import InstinctModel, answer

model = InstinctModel.from_pretrained("instinct-tuned-4b")  # or "reference-qwen3.8-27b"
result = answer(model, {
    "state": {"customer": "My package says delivered but it is not here."},
    "questions": {
        "escalate": {"type": "noul", "instructions": "The customer asks for a human agent."},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["Not urgent", "Somewhat urgent", "Urgent", "Critical"]},
    },
})
# {"answers": {"escalate": {"type": "noul", "noul": <P(yes)>},
#              "urgency": {"type": "score", "probabilities": {"0": ..., "3": ...}, "score": <expected level>}},
#  "usage": {"input_tokens": ..., "output_tokens": 0}}
```

For single records in the lower-level format (`id`, `group_id`, `state`, `instructions`, `primitive`, `criteria`), call `model.decide(record)`. It returns probabilities, the prediction, raw logits and the prompt token count.

### Request format

| Field | Type | Notes |
|---|---|---|
| `state` | string, object or array | Objects and arrays are rendered as `json.dumps(..., ensure_ascii=False)`. |
| `questions` | object | Maps each question id to a question. |
| `questions.*.type` | `noul`, `choice` or `score` | |
| `questions.*.instructions` | string | The question or proposition. |
| `questions.*.criteria` | depends on type | `choice`: an object mapping 2–16 option keys to descriptions. `score`: a list of 2–16 ordered level descriptions. `noul`: optional `{"true", "false"}` wording; whether the model sees it depends on the recipe. |
| `model`, `permutations` | optional | `model` must match the served model. `permutations` must be 1 or omitted; each recipe fixes its own option orders. |

### How the runtime scores a question

1. The recipe renders the question into one prompt per option order and assigns a label token to each candidate.
2. One forward pass per order reads the label-token logits at the last position. `instinct-tuned-4b` applies only the candidate LM-head rows to the full-depth, final-norm hidden state; `reference-qwen3.8-27b` reads the full head.
3. Each order's logits are divided by the recipe's temperature and softmaxed, mapped back to candidate ids, and averaged over orders.

Each model's README describes its prompt exactly; `instinct/prompt.py` defines the `instinct-tuned-4b` prompt.

## Optimized serving

`serving/vllm/` holds the optional vLLM patch and launcher used for the `reference-qwen3.8-27b` baseline. The Transformers path in `instinct/` is the reference; the canaries record how closely the two agree.

## Test

```bash
pip install -e ".[test]" && pytest -q     # CPU only
```

## About

Instinct is developed by [ZooWork](https://zoowork.ai), which turns your expertise into an AI agent in minutes. Product page, demos and updates: [instinct.zoowork.ai](https://instinct.zoowork.ai/).

## License

This code is released under Apache-2.0 (see `LICENSE` and `NOTICE`). `serving/vllm/` contains modified vLLM files, which carry vLLM's Apache-2.0 license and notice (`serving/vllm/LICENSE`, `serving/vllm/NOTICE`). Model weights carry their own license; see the model card.
