# Adding a model

Each released model gets one directory, `models/<name>/`, laid out like `models/instinct-tuned-4b/`:

| File | Content |
|---|---|
| `README.md` | what the model is, its HF link, the run command and a results table |
| `model.json` | `name`, `hf_repo`, `base_model`, `prompt_version`, `temperature`, `full_depth`, `question_types`, `reference_hardware` |
| `examples/request.json`, `examples/expected.json` | an example request and the model's answer on the reference hardware |
| `canary/records.jsonl` or `canary/requests.jsonl`, plus `canary/reference.jsonl` | canary inputs (records or SystemOne requests) and the logits or answers production returned for them; set `canary_tolerance` in `model.json` |

The HF repo must contain a `decision_config.json` with `prompt_version`, `temperature` and `full_depth`. The runtime reads the serving temperature from that file.

The shared runtime supports Qwen3.5 models that use the `jev.dynamic.prompt.v2` contract and a full-depth candidate-row readout. A model with a different prompt contract, readout depth or architecture needs a runtime change first. Add a test for that change in `tests/`.

`tests/test_models.py` checks every `models/*/` directory automatically.
