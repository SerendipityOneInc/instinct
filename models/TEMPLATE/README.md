# Adding a model

Each released model gets one directory, `models/<name>/`, laid out like `models/instinct-tuned-4b/`:

| File | Content |
|---|---|
| `README.md` | what the model is, its HF link, the run command and a results table |
| `model.json` | `name`, `hf_repo`, `base_model`, `prompt_version`, `temperature`, `full_depth`, `question_types`, `reference_hardware`; optional `temperature_by_type` |
| `examples/request.json`, `examples/expected.json` | an example request and the model's answer on the reference hardware |
| `canary/records.jsonl` or `canary/requests.jsonl`, plus `canary/reference.jsonl` | canary inputs (records or SystemOne requests) and the logits or answers production returned for them; set `canary_tolerance` in `model.json` |

The HF repo must contain a `decision_config.json` with `prompt_version`, `temperature` and `full_depth`, plus `temperature_by_type` when the recipe sets one. `temperature` is the serving temperature; `temperature_by_type` optionally overrides it per question type (`noul`, `choice`, `score`), for example `{"noul": 0.5}`. The runtime refuses to load weights whose `decision_config.json` temperatures differ from the recipe's (a missing `temperature_by_type` means none).

The shared runtime supports Qwen3.5 models that use the `instinct.prompt.v1` prompt (`instinct/prompt.py`) and a full-depth candidate-row readout. A model with a different prompt contract, readout depth or architecture needs a runtime change first. Add a test for that change in `tests/`.

`tests/test_models.py` checks every `models/*/` and `reference/*/` directory automatically.
