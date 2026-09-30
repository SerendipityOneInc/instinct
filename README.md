# ZooWork - Instinct model

[![ZooWork](https://img.shields.io/badge/built%20by-ZooWork-6C47FF?style=flat-square)](https://zoowork.ai)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-3DA639?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Tests](https://github.com/SerendipityOneInc/instinct/actions/workflows/test.yml/badge.svg)](https://github.com/SerendipityOneInc/instinct/actions/workflows/test.yml)
[![Views](https://hits.sh/github.com/SerendipityOneInc/instinct.svg?style=flat-square&label=views)](https://hits.sh/github.com/SerendipityOneInc/instinct/)

**An open decision-model family from the [ZooWork](https://zoowork.ai) team at Serendipity One Inc.**

[Models on Hugging Face](https://huggingface.co/srpone) · [Hosted API](https://instinct.zoowork.ai/) · [API documentation](https://instinct.zoowork.ai/docs/)

Instinct turns shared context and typed questions into distributions over fixed candidates. It supports binary judgments, categorical choices and ordered scores without generating text. All three models are open on Hugging Face, share the same typed API, and return zero generated output tokens.

## Model family

<table>
  <thead>
    <tr>
      <th width="25%">Model</th>
      <th width="27%">Training</th>
      <th width="30%">Inference</th>
      <th width="18%">Hosted API</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong><code>instinct</code></strong><br><a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen base weights</a></td>
      <td>No additional training · frozen Qwen3.8-27B</td>
      <td>Single-order logit readout · compiled serving</td>
      <td><a href="https://instinct.zoowork.ai/docs/#models">Use via API</a></td>
    </tr>
    <tr>
      <td><strong><code>instinct-dual-4b</code></strong><br><a href="https://huggingface.co/Qwen/Qwen3.5-4B">Qwen base weights</a></td>
      <td>No additional training · frozen Qwen3.5-4B Instruct</td>
      <td>Dual-order averaging · concurrent serving</td>
      <td><a href="https://instinct.zoowork.ai/docs/#models">Use via API</a></td>
    </tr>
    <tr>
      <td><strong><code>instinct-tuned-4b</code></strong><br><a href="https://huggingface.co/srpone/instinct-tuned-4b">ZooWork tuned weights</a></td>
      <td>Decision-task fine-tune of Qwen3.5-4B</td>
      <td>Single-order logit readout</td>
      <td><a href="https://instinct.zoowork.ai/docs/#models">Use via API</a></td>
    </tr>
  </tbody>
</table>

The first two models use unchanged Qwen weights and add inference-time decision readouts. `instinct-tuned-4b` publishes ZooWork's fine-tuned checkpoint. This GitHub repository provides the shared request contract, reference runtime, examples and serving utilities.

## Capabilities

- **Noul** — a yes/no proposition returned as `P(yes)` in `[0, 1]`.
- **Choice** — a distribution over 2–16 named candidates, plus the selected candidate and confidence.
- **Score** — a distribution over 2–16 ordered levels and their probability-weighted expected level.
- **Shared state** — ask several independent questions about the same string, object or array in one request.
- **Zero text generation** — responses contain structured answers and report `output_tokens: 0`.
- **Inspectable outputs** — the lower-level API exposes probabilities, predictions, raw candidate logits and prompt-token counts.

## Measured results

Reporting-only results on all 231 published JevBench tasks:

| Production model | Correct | Accuracy | Local p50 | Local p95 |
|---|---:|---:|---:|---:|
| **`instinct`** | **202/231** | **87.45%** | 36.6 ms | 243.2 ms |
| **`instinct-tuned-4b`** | **198/231** | **85.71%** | 62 ms | ≈110 ms (est.) |
| **`instinct-dual-4b`** | **190/231** | **82.25%** | 40.9 ms | 79.7 ms |

All reported runs completed 231/231 items. These are public-set diagnostics—not held-out scores, official full-suite ranks, or full JevBench Intelligence and Calibration scores. The public subset has no judge tier and was consulted during development.

Latency is warmed, serial serving latency excluding public Internet, TLS and gateway overhead; the 27B and dual-4B values pool three complete passes, while tuned-4B's p95 is an estimate from its measured 62 ms direct-upstream p50 and observed end-to-end tail spread, not an SLO.

For the open tuned model, the split is easy 48/48, standard 69/72 and hard 81/111. See its [release notes](models/instinct-tuned-4b/README.md) and the [27B recipe notes](reference/qwen3.8-27b/README.md) for artifact-specific conditions.

## Installation

Python 3.10 or newer and a CUDA GPU with BF16 support (Ampere or newer) are required for inference. `instinct-tuned-4b` downloads about 8.5 GB of weights from Hugging Face on first use and runs on a single 24 GB GPU; the 27B reference needs about 54 GB (see [`reference/qwen3.8-27b`](reference/qwen3.8-27b/README.md)). The `flash-linear-attention` Triton kernels are compiled on first use, so the environment also needs a C compiler; slim images without `gcc` (for example the `pytorch/pytorch:*-runtime` tags) install fine but fail on the first forward pass with `Failed to find C compiler`.

The simplest environment is an [NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), which ships a matched torch, CUDA and compiler. Install on top of it and keep its torch: the `gpu` extra accepts NGC's `2.8.0a0` builds, so pip leaves them in place.

```bash
docker run --gpus all --shm-size=8g -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:25.06-py3
```

| Environment | GPU | Result |
|---|---|---|
| torch 2.13.0+cu129, triton 3.7.1 (reference) | H200 | canary logits match production exactly |
| NGC `pytorch:25.06-py3` (torch 2.8.0a0, CUDA 12.9) | A10 | all tests pass; CLI, Python and HTTP examples run as written; canary predictions 14/14, probabilities within 0.021 |

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

To reproduce the frozen 27B `instinct` recipe locally:

```bash
instinct-decide --model reference-qwen3.8-27b \
  reference/qwen3.8-27b/examples/request.json
```

The first run downloads the weights; later runs load them from the Hugging Face cache. The output for the example request is checked in as [`examples/expected.json`](models/instinct-tuned-4b/examples/expected.json), produced on an H200. Other GPUs give the same choices with probabilities that can differ in the second decimal place (see the canary note in the [model README](models/instinct-tuned-4b/README.md)).

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

### Hosted API pricing

| Model ID | Input / 1M tokens | Output |
|---|---:|---:|
| `instinct` | $0.03 | Free |
| `instinct-dual-4b` | $0.01 | Free |
| `instinct-tuned-4b` | $0.01 | Free |

These are serving prices for ZooWork's managed API, not licenses or usage fees for the open weights. Running the models yourself does not use the hosted API billing plan.

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

The hosted product offers all three model IDs behind the same typed API. The repository artifacts currently cover tuned-4B and the frozen 27B recipe as described above; consult the live API documentation for current availability and pricing.

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
5. applies the recipe temperature (per question type when the recipe sets `temperature_by_type`) and a softmax;
6. maps probabilities back to the caller's candidate IDs and averages configured orders.

`instinct-tuned-4b` uses one option order and applies only the candidate LM-head rows to the full-depth, final-norm hidden state. Its temperature is 2.80 for `choice` and `score` and 0.50 for `noul`; the lower `noul` temperature sharpens P(yes) without changing any prediction. The prompt bytes are part of the trained model contract: changing [`instinct/prompt.py`](instinct/prompt.py) can change the model's output.

## Testing and contributing

```bash
pip install -e ".[test]"
pytest -q
```

Issues and pull requests are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md). Changes to prompt rendering require matching canary evidence. Report security issues privately through [GitHub security advisories](https://github.com/SerendipityOneInc/instinct/security/advisories/new); see [`SECURITY.md`](SECURITY.md).

## Citation

```bibtex
@software{instinct2026,
  title   = {Instinct: An Open Decision-Model Family},
  author  = {{ZooWork Team, Serendipity One Inc.}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/SerendipityOneInc/instinct},
  note    = {Models: instinct, instinct-dual-4b, instinct-tuned-4b}
}
```

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). Copyright 2026 Serendipity One Inc. See [`NOTICE`](NOTICE).

The vLLM-derived serving files are also Apache-2.0; see [`serving/vllm/LICENSE`](serving/vllm/LICENSE) and [`serving/vllm/NOTICE`](serving/vllm/NOTICE). Model weights are distributed separately under the license stated on their respective model cards.

## References

- [Reflex](https://github.com/kshetrajna12/reflex) — an open implementation of typed, candidate-scored decision models.
- [JevBench](https://github.com/fstandhartinger/jevbench) — the public benchmark used for the diagnostic results above.
- [Qwen](https://github.com/QwenLM/Qwen3) — the open model family behind the Instinct models.
- [vLLM](https://github.com/vllm-project/vllm) — the serving engine used by the optional optimized path.
