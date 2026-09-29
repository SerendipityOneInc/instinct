# Instinct

[![ZooWork](https://img.shields.io/badge/built%20by-ZooWork-6C47FF?style=flat-square)](https://zoowork.ai)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-3DA639?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Tests](https://github.com/SerendipityOneInc/instinct/actions/workflows/test.yml/badge.svg)](https://github.com/SerendipityOneInc/instinct/actions/workflows/test.yml)
[![Views](https://hits.sh/github.com/SerendipityOneInc/instinct.svg?style=flat-square&label=views)](https://hits.sh/github.com/SerendipityOneInc/instinct/)

**An open decision model from the [ZooWork](https://zoowork.ai) team at Serendipity One Inc.**

[Hosted API](https://instinct.zoowork.ai/) · [API documentation](https://instinct.zoowork.ai/docs/) · [Model weights](https://huggingface.co/srpone/instinct-tuned-4b)

Instinct turns shared context and typed questions into distributions over fixed candidates. It supports binary judgments, categorical choices and ordered scores without generating text. Each question is answered by reading candidate-token logits from a single forward pass per option order.

This repository releases **`instinct-tuned-4b`**, a post-trained Qwen3.5-4B decision model, together with its reference runtime, examples and reproducibility canaries. It also contains **`reference-qwen3.8-27b`**, an untrained comparison baseline—not a second released Instinct model.

## Model release

| | `instinct-tuned-4b` | `reference-qwen3.8-27b` |
|---|---|---|
| Status | **Released Instinct model** | Reference baseline only |
| Directory | [`models/instinct-tuned-4b`](models/instinct-tuned-4b) | [`reference/qwen3.8-27b`](reference/qwen3.8-27b) |
| Weights | [`srpone/instinct-tuned-4b`](https://huggingface.co/srpone/instinct-tuned-4b) | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Pinned revision | `50546bf18c4f11115dc92bc1ce82c75cfc2968db` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Training | Merged LoRA fine-tune of Qwen3.5-4B | Frozen, untrained Qwen3.8-27B |
| Prompt | `instinct.prompt.v1` | jqv prompt v1, hash `4f85a0b34776` |
| Readout | Candidate LM-head rows at full depth | Full-head option-letter logits |
| Orders / temperature | 1 / 2.80 calibrated | 1 / 1.0 uncalibrated |
| Prompt limit | 8,192 tokens, no truncation | 32,768 tokens, no truncation |
| Question types | `noul`, `choice`, `score` | `noul`, `choice`, `score` |
| Reference hardware | 1× H200, BF16, Transformers 5.16.1 | 1× H200, BF16, Transformers 5.16.1 |

A model is defined by a recipe in [`instinct/recipes/`](instinct/recipes/): pinned weights, prompt, candidate labels, readout, option orders and temperature. Each model directory includes metadata, an example request and expected response, and canary outputs captured from the reference deployment.

## Capabilities

- **Noul** — a yes/no proposition returned as `P(yes)` in `[0, 1]`.
- **Choice** — a distribution over 2–16 named candidates, plus the selected candidate and confidence.
- **Score** — a distribution over 2–16 ordered levels and their probability-weighted expected level.
- **Shared state** — ask several independent questions about the same string, object or array in one request.
- **Zero text generation** — responses contain structured answers and report `output_tokens: 0`.
- **Inspectable outputs** — the lower-level API exposes probabilities, predictions, raw candidate logits and prompt-token counts.

## Measured results

JevBench public subset, all 231 published tasks, BF16 on one NVIDIA H200:

| Split | `instinct-tuned-4b` | `reference-qwen3.8-27b` |
|---|---:|---:|
| Easy | 48/48 (100.00%) | 48/48 (100.00%) |
| Standard | 69/72 (95.83%) | 69/72 (95.83%) |
| Hard | 81/111 (72.97%) | 84/111 (75.68%) |
| **Total** | **198/231 (85.71%)** | **201/231 (87.01%)** |

Both runs evaluated 231/231 items. These are public-set diagnostics, not held-out scores, official full-suite ranks, or full JevBench Intelligence and Calibration scores: the public subset has no judge tier, and those axes require additional data. The public items were used for model selection during development. The tuned model was scored with the official JevBench client in the original option order; the baseline result comes from our evaluation harness. Their probability scales also differ: the tuned model uses `T=2.80`, while the baseline is uncalibrated at `T=1.0`.

See the [`instinct-tuned-4b` release notes](models/instinct-tuned-4b/README.md) and [reference baseline notes](reference/qwen3.8-27b/README.md) for the exact evaluation conditions.

## Installation

Python 3.10 or newer and a CUDA GPU are required for inference. The verified reference environment uses one H200 in BF16.

```bash
git clone https://github.com/SerendipityOneInc/instinct.git
cd instinct
pip install -e ".[gpu]"
```

Layer execution is pinned to `transformers==5.16.1`. The runtime refuses other versions unless `INSTINCT_ALLOW_UNVERIFIED_TRANSFORMERS=1` is set. `flash-linear-attention==0.5.2` and `fla-core==0.5.2` provide the Qwen3.5 linear-attention kernels.

## Local inference

### CLI

```bash
instinct-decide --model instinct-tuned-4b \
  models/instinct-tuned-4b/examples/request.json
```

To run the comparison baseline instead:

```bash
instinct-decide --model reference-qwen3.8-27b \
  reference/qwen3.8-27b/examples/request.json
```

Use `--weights <local-directory-or-hf-repo>` to override the recipe's weight location.

### Python

```python
from instinct import InstinctModel, answer

model = InstinctModel.from_pretrained("instinct-tuned-4b")
result = answer(model, {
    "state": {"customer": "My package says delivered but it is not here."},
    "questions": {
        "escalate": {
            "type": "noul",
            "instructions": "Is human support required?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["Not urgent", "Somewhat urgent", "Urgent", "Critical"],
        },
    },
})

print(result["answers"])
```

### Reference HTTP server

```bash
instinct-serve --model instinct-tuned-4b --port 8008
curl -s http://127.0.0.1:8008/v1/systemone \
  -H 'Content-Type: application/json' \
  --data @models/instinct-tuned-4b/examples/request.json
```

The local server exposes `POST /v1/systemone` and `GET /health`, binds to `127.0.0.1` by default, and serves one model request at a time. It is a reference server without authentication, TLS or rate limiting; do not expose it directly to an untrusted network. Request bodies are limited to 8 MiB and 64 questions.

## Hosted API

ZooWork operates the production Instinct API. Create an API key and explore the model family at [instinct.zoowork.ai](https://instinct.zoowork.ai/); the current authentication, model and usage contract is documented at [instinct.zoowork.ai/docs](https://instinct.zoowork.ai/docs/).

```bash
export INSTINCT_API_KEY='your-key'

curl https://api.zoowork.ai/v1/systemone \
  -H "Authorization: Bearer ${INSTINCT_API_KEY}" \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "instinct-tuned-4b",
    "state": "The payment service is down for all customers.",
    "questions": {
      "outage": {
        "type": "noul",
        "instructions": "Is the payment service experiencing an outage?"
      },
      "team": {
        "type": "choice",
        "instructions": "Which team should respond?",
        "criteria": {
          "engineering": "Technical service failures",
          "billing": "Invoice questions"
        }
      }
    }
  }'
```

The hosted product offers additional model IDs behind the same typed API. This repository's open-model release remains `instinct-tuned-4b`; consult the live API documentation for hosted availability and pricing.

## Request and response contract

### Request

| Field | Type | Description |
|---|---|---|
| `state` | non-empty string, object or array | Shared context for every question. Objects and arrays are rendered as JSON. |
| `questions` | object | Maps one or more question IDs to question definitions. |
| `questions.*.type` | `noul`, `choice`, `score` | Selects the answer shape. |
| `questions.*.instructions` | non-empty string | The proposition or question. |
| `questions.*.criteria` | type-dependent | Choice map, ordered score levels, or optional Noul wording. |
| `model` | optional string | Must match the model served by the endpoint. |
| `permutations` | optional integer | Must be `1` or omitted; each recipe fixes its own option orders. |

Criteria by question type:

- `noul`: optional `{"true": "...", "false": "..."}` descriptions;
- `choice`: an object mapping 2–16 candidate IDs to descriptions;
- `score`: a list of 2–16 ordered level descriptions.

Unknown request or question fields are rejected.

### Response

```json
{
  "answers": {
    "outage": {
      "type": "noul",
      "noul": 0.98
    },
    "team": {
      "type": "choice",
      "choice": "engineering",
      "probabilities": {
        "engineering": 0.97,
        "billing": 0.03
      },
      "confidence": 0.81
    }
  },
  "usage": {
    "input_tokens": 312,
    "output_tokens": 0
  }
}
```

The values above illustrate the response shape; exact probabilities and token counts depend on the request and runtime. Noul returns `P(yes)`. Choice returns the arg-max candidate, the complete candidate distribution and confidence. Score returns the level distribution and its expected zero-based level. Probabilities are rounded to six decimal places.

## How it works

For each question, the selected recipe:

1. renders the shared state, instructions and candidate rubric into a deterministic prompt;
2. assigns one single-token label (`A`, `B`, …) to each candidate;
3. runs the model once per configured option order;
4. reads the candidate-label logits at the final prompt position;
5. applies the recipe temperature and a softmax;
6. maps probabilities back to the caller's candidate IDs and averages configured orders.

`instinct-tuned-4b` uses one option order and applies only the candidate LM-head rows to the full-depth, final-norm hidden state. The prompt bytes are part of the trained model contract: changing [`instinct/prompt.py`](instinct/prompt.py) can change the model's output.

## Reproducibility

Weights and runtime dependencies are pinned in the recipes and package metadata. Verify an installation against the recorded reference outputs:

```bash
python scripts/check_canaries.py --model instinct-tuned-4b
python scripts/check_canaries.py --model reference-qwen3.8-27b
```

- `instinct-tuned-4b`: 14 synthetic requests; predictions must agree and every probability must remain within `0.02` after the `T=2.80` softmax.
- `reference-qwen3.8-27b`: 15 requests containing 18 questions; predictions must agree and probabilities may differ by at most `0.05` across the Transformers and patched-vLLM BF16 paths.

An inexact raw-logit match alone is not a failure because GPU kernels can shift BF16 logits slightly. [`scripts/parity_check.py`](scripts/parity_check.py) provides item-level logit comparison, while [`scripts/compare_endpoint.py`](scripts/compare_endpoint.py) compares a local model with a running System One endpoint.

The optional [`serving/vllm`](serving/vllm) path pins vLLM 0.17.1 and implements native prefill-only candidate readout for the 27B reference baseline. The Transformers path under `instinct/` remains the reference implementation.

## Limitations

- Public JevBench results are development diagnostics, not held-out evidence.
- Only one H200/BF16 reference setup is verified; other GPUs, kernels and dtypes may shift probabilities.
- Local inference requires a CUDA GPU. Only the test suite is CPU-only.
- Inputs over the recipe's prompt limit are rejected rather than truncated.
- Both included recipes use one option order. The 27B baseline probabilities are uncalibrated.
- The bundled HTTP server is intended for local reference use, not direct production exposure.
- See the [model card](https://huggingface.co/srpone/instinct-tuned-4b) for training details and model-specific limitations.

## Repository layout

```text
instinct/                    reference runtime and model recipes
models/instinct-tuned-4b/    released model metadata, examples and canaries
reference/qwen3.8-27b/       untrained comparison baseline
scripts/                     parity, canary and endpoint checks
serving/vllm/                optional Apache-2.0 vLLM serving path
tests/                       CPU test suite
```

## Testing and contributing

```bash
pip install -e ".[test]"
pytest -q
```

Issues and pull requests are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md). Changes to prompt rendering require matching canary evidence. Report security issues privately through [GitHub security advisories](https://github.com/SerendipityOneInc/instinct/security/advisories/new); see [`SECURITY.md`](SECURITY.md).

## Citation

```bibtex
@software{instinct2026,
  title   = {Instinct: An Open Decision Model},
  author  = {{ZooWork Team, Serendipity One Inc.}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/SerendipityOneInc/instinct},
  note    = {Model: srpone/instinct-tuned-4b}
}
```

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). Copyright 2026 Serendipity One Inc. See [`NOTICE`](NOTICE).

The vLLM-derived serving files are also Apache-2.0; see [`serving/vllm/LICENSE`](serving/vllm/LICENSE) and [`serving/vllm/NOTICE`](serving/vllm/NOTICE). Model weights are distributed separately under the license stated on their respective model cards.

## References

- [Reflex](https://github.com/kshetrajna12/reflex) — an open implementation of typed, candidate-scored decision models.
- [JevBench](https://github.com/fstandhartinger/jevbench) — the public benchmark used for the diagnostic results above.
- [Qwen](https://github.com/QwenLM/Qwen3) — the open model family behind the released model and reference baseline.
- [vLLM](https://github.com/vllm-project/vllm) — the serving engine used by the optional optimized path.
