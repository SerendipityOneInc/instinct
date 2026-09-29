# Instinct

Instinct models are **decision models**. Given a shared state and a question with fixed candidates, they return a probability per candidate from one forward pass per option order, with no text generation.

This repository holds one shared runtime, the `instinct` Python package, and one directory per released model. Each model is a *recipe* (`instinct/recipes/`): its weights, prompt, label tokens, readout, option orders and temperature. Two of the three models use the untrained Qwen weights directly; their recipes pin the Qwen repository and revision.

## Models

| Model | Directory | Weights | Orders | T | Question types |
|---|---|---|---|---|---|
| instinct-tuned-4b | [`models/instinct-tuned-4b`](models/instinct-tuned-4b) | [`<hf-org>/instinct-tuned-4b`](https://huggingface.co/<hf-org>/instinct-tuned-4b) (LoRA fine-tune of Qwen3.5-4B, merged) | 1 | 2.80 (calibrated) | noul, choice, score |
| instinct | [`models/instinct`](models/instinct) | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) @ `1d4bf0f`, untrained | 1 | 1 (uncalibrated) | noul, choice, score |
| instinct-dual-4b | [`models/instinct-dual-4b`](models/instinct-dual-4b) | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) @ `851bf6e`, untrained | 2 | 1 (uncalibrated) | noul, choice, score |

The HF repositories `<hf-org>/instinct` and `<hf-org>/instinct-dual-4b` hold model cards only.

Each model directory contains:

- a README with a results table;
- `model.json` metadata;
- an example request with its expected answer;
- canary requests with the outputs our production deployment returned, and a tolerance (`scripts/check_canaries.py --model <name>`).

To add a model, see [`models/TEMPLATE`](models/TEMPLATE).

## Install

A CUDA GPU is required. The runtime is verified on one H200 in bf16.

```bash
git clone https://github.com/SerendipityOneInc/instinct && cd instinct
pip install -e ".[gpu]"
```

Layer execution is verified against `transformers==5.16.1`, and the runtime refuses to start with any other version unless `INSTINCT_ALLOW_UNVERIFIED_TRANSFORMERS=1` is set. `flash-linear-attention` provides the Qwen3.5 linear-attention kernels.

## Use

```bash
instinct-decide --model instinct-tuned-4b models/instinct-tuned-4b/examples/request.json
instinct-serve  --model instinct-tuned-4b --port 8008
# --weights <local dir or HF repo> overrides where the weights come from
curl -s localhost:8008/v1/systemone -d @models/instinct-tuned-4b/examples/request.json
```

```python
from instinct import InstinctModel, answer

model = InstinctModel.from_pretrained("instinct-tuned-4b")
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
2. One forward pass per order reads the label-token logits at the readout position. `instinct-tuned-4b` applies only the candidate LM-head rows to the final-norm hidden state; the other two read the full head at the last position.
3. Each order's logits are divided by the recipe's temperature and softmaxed, mapped back to candidate ids, and averaged over orders.

Each model's README describes its prompt exactly.

## Optimized serving

`serving/vllm/` holds the optional vLLM patches and launchers our production uses for `instinct` and `instinct-dual-4b`. The Transformers path in `instinct/` is the reference; the canaries record how closely the two agree.

## Test

```bash
pip install -e ".[test]" && pytest -q     # CPU only
```

## License

This code is released under Apache-2.0 (see `LICENSE`). `instinct/_vendor/apus_runtime/` holds MIT-licensed code from [APUS-OpenJev-v1](https://huggingface.co/apus-ailab/APUS-OpenJev-v1); `instinct/_vendor/reflex/` holds MIT-licensed code from [Reflex](https://github.com/kshetrajna12/reflex). Each vendored directory's `LICENSE` and `SOURCE.txt` record the origin and changes. Each model's weights carry their own license; see its model card.
