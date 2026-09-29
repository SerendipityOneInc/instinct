# Adding a model

A model is a recipe plus a directory:

1. **Recipe** — `instinct/recipes/<name>.py` defines a `Recipe` (see `instinct/recipes/base.py`): the weights repo and pinned revision, the prompt encoding and label tokens (`encode`), the option orders (`orders`), the readout (`candidate_rows` or `full_head`), the temperature and the input limit. Register it in `instinct/recipes/__init__.py` and add tests in `tests/`.
2. **Directory** — `models/<name>/`:

| File | Content |
|---|---|
| `README.md` | what the model is, its weights, how its prompt is built, the run command, results |
| `model.json` | `name`, `hf_repo`, `base_model`, `prompt_version`, `temperature`, `full_depth`, `question_types`, `reference_hardware`, `canary_tolerance` |
| `examples/request.json`, `examples/expected.json` | an example request and the model's answer on the reference hardware |
| `canary/records.jsonl` or `canary/requests.jsonl`, plus `canary/reference.jsonl` | canary inputs (records or SystemOne requests) and the logits or answers the reference deployment returned for them |

A weights repo may contain a `decision_config.json`. It is optional; when present, the runtime refuses to load if its `temperature` differs from the recipe's.

`tests/test_models.py` checks every `models/*/` directory against the recipe registry. `scripts/check_canaries.py --model <name>` checks an install against the canaries.
