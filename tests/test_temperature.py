"""Per-question-type temperature (Recipe.temperature_by_type), CPU only."""

import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import pytest

from instinct.model import check_decision_config
from instinct.recipes import get
from instinct.recipes.base import Encoded
from instinct.recipes.jqv_27b import RECIPE as JQV

ROOT = Path(__file__).parents[1]
TUNED = get("instinct-tuned-4b")

BODY = {
    "state": "The payment service is down.",
    "questions": {
        "outage": {"type": "noul", "instructions": "Is there an outage?"},
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"eng": "Engineering", "billing": "Billing"}},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["Low", "Medium", "High"]},
    },
}


class CharTokenizer:
    all_special_ids = [0]

    def apply_chat_template(self, messages, **kwargs):
        return "<u>" + messages[0]["content"] + "</u><a>"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class JqvTokenizer:
    """One id per character; " X" is a single token (as in test_recipe_jqv)."""

    all_special_ids = [0]

    def encode(self, text, add_special_tokens=False):
        if len(text) == 2 and text[0] == " " and text[1].isalpha():
            return [1000 + ord(text[1])]
        return [ord(c) for c in text]


def softmax(logits, t):
    e = [math.exp((x - max(logits)) / t) for x in logits]
    return [v / sum(e) for v in e]


def fake_logits(n):
    return [1.0] + [0.0] * (n - 1)


def test_tuned_4b_temperatures():
    assert TUNED.temperature == 2.8 and TUNED.temperature_by_type == {"noul": 0.5}
    assert TUNED.temperature_for("noul") == 0.5
    assert TUNED.temperature_for("choice") == TUNED.temperature_for("score") == 2.8
    with pytest.raises(ValueError):
        TUNED.temperature_for("score_level")


def test_reference_27b_unchanged():
    assert JQV.temperature == 1.0 and JQV.temperature_by_type == {}
    assert all(JQV.temperature_for(k) == 1.0 for k in ("noul", "choice", "score"))


@pytest.mark.parametrize("mapping", [{"yesno": 1.0}, {"score_level": 1.0}, {"noul": 0.0},
                                     {"noul": -1.0}, {"noul": math.inf}, {"noul": math.nan},
                                     {"noul": True}, {"noul": "0.5"}])
def test_validation_rejects(mapping):
    with pytest.raises(ValueError):
        dataclasses.replace(TUNED, temperature_by_type=mapping)


def test_records_carry_question_type_and_score_is_not_choice():
    records = TUNED.records(BODY)
    assert [(k, r["question_type"], r["primitive"]) for _, k, r in records] == [
        ("noul", "noul", "noul"), ("choice", "choice", "choice"), ("score", "score", "choice")]
    assert [TUNED.record_temperature(r) for _, _, r in records] == [0.5, 2.8, 2.8]
    # Score keeps its own key even though its primitive is "choice".
    score_only = dataclasses.replace(TUNED, temperature_by_type={"score": 1.5})
    assert [score_only.record_temperature(r) for _, _, r in records] == [2.8, 2.8, 1.5]


@pytest.mark.parametrize("recipe", [TUNED, JQV], ids=lambda r: r.name)
def test_question_type_does_not_change_prompt_bytes(recipe):
    for _, _, record in recipe.records(BODY):
        bare = {k: v for k, v in record.items() if k != "question_type"}
        (order,) = recipe.orders(record)
        tok = JqvTokenizer() if recipe is JQV else CharTokenizer()
        assert recipe.encode(tok, record, order) == recipe.encode(tok, bare, order)


def test_transformers_path_merge_uses_type_temperature():
    """InstinctModel.decide -> recipe.merge on records from recipe.records."""
    for _, kind, record in TUNED.records(BODY):
        ids = [c["id"] for c in record["criteria"]]
        logits = fake_logits(len(ids))
        probs = TUNED.merge(record, [Encoded([1], list(range(len(ids))), ids)], [logits])
        expected_t = 0.5 if kind == "noul" else 2.8
        assert probs == pytest.approx(softmax(logits, expected_t), abs=1e-15)
        assert probs.index(max(probs)) == 0  # prediction independent of temperature


def test_canary_reference_matches_recipe():
    """Reference probabilities equal recipe.merge(reference logits) for every canary row."""
    base = ROOT / "models/instinct-tuned-4b/canary"
    reference = {json.loads(l)["id"]: json.loads(l) for l in open(base / "reference.jsonl")}
    for line in open(base / "records.jsonl"):
        record = json.loads(line)
        ids = [c["id"] for c in record["criteria"]]
        row = reference[record["id"]]
        probs = TUNED.merge(record, [Encoded([1], list(range(len(ids))), ids)], [row["logits"]])
        assert probs == pytest.approx([row["probabilities"][i] for i in ids], abs=1e-12)
        assert ids[probs.index(max(probs))] == row["prediction"]


def _vllm_readout():
    path = ROOT / "serving/vllm/common/vllm_readout.py"
    spec = importlib.util.spec_from_file_location("vllm_readout_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vllm_readout_path_uses_type_temperature(monkeypatch):
    readout = _vllm_readout()
    model = readout.VLLMDecisionModel(CharTokenizer(), TUNED, primary=None)
    monkeypatch.setattr(model, "_score", lambda enc: [fake_logits(len(e.label_token_ids)) for e in enc])
    out = readout.answer_many(model, BODY)["answers"]
    assert out["outage"]["noul"] == round(softmax([1.0, 0.0], 0.5)[0], 6)
    assert out["team"]["probabilities"]["eng"] == round(softmax([1.0, 0.0], 2.8)[0], 6)
    assert out["urgency"]["probabilities"]["0"] == round(softmax([1.0, 0.0, 0.0], 2.8)[0], 6)
    results = model.decide_many([r for _, _, r in TUNED.records(BODY)])
    assert [r["temperature"] for r in results] == [0.5, 2.8, 2.8]


DECLARED = {"prompt_version": "instinct.prompt.v1", "temperature": 2.8,
            "temperature_by_type": {"noul": 0.5}, "full_depth": 32}


def test_decision_config_matches():
    check_decision_config(DECLARED, TUNED)
    check_decision_config({"temperature": 1.0}, JQV)  # missing mapping == empty


@pytest.mark.parametrize("declared, recipe", [
    ({**DECLARED, "temperature_by_type": {"noul": 0.4}}, TUNED),
    ({**DECLARED, "temperature_by_type": {}}, TUNED),
    ({k: v for k, v in DECLARED.items() if k != "temperature_by_type"}, TUNED),
    ({**DECLARED, "temperature_by_type": {"noul": 0.5, "score": 2.8}}, TUNED),
    ({**DECLARED, "temperature": 2.0}, TUNED),
    ({"temperature": 1.0, "temperature_by_type": {"noul": 0.5}}, JQV),
])
def test_decision_config_mismatch_raises(declared, recipe):
    with pytest.raises(ValueError):
        check_decision_config(declared, recipe)
