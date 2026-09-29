import json
import math
import types

import pytest

from instinct import calibrate
from instinct.systemone import answer, confidence, to_answer, to_records


def body(**questions):
    return {"state": "The server is offline.", "questions": questions}


def test_calibrate_is_temperature_softmax():
    probs = calibrate([2.0, 0.0], 2.0)
    assert math.isclose(probs[0], 1 / (1 + math.exp(-1)))
    assert math.isclose(sum(probs), 1.0)
    with pytest.raises(ValueError):
        calibrate([1.0], 0.0)


def test_noul_uses_canonical_criteria_even_with_custom_wording():
    (_, kind, record), = to_records(body(q={
        "type": "noul", "instructions": "The server is offline.",
        "criteria": {"true": "It is down", "false": "It is up"}}))
    assert kind == "noul"
    assert record["primitive"] == "noul"
    assert [c["id"] for c in record["criteria"]] == ["yes", "no"]
    assert record["criteria"][0]["description"] == "The stated proposition is true."


def test_score_levels_become_ordered_choices():
    (_, kind, record), = to_records(body(q={
        "type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]}))
    assert kind == "score"
    assert record["primitive"] == "choice"
    assert [c["id"] for c in record["criteria"]] == ["0", "1", "2"]


def test_structured_state_is_rendered_as_json_text():
    request = {"state": {"msg": "café"}, "questions": {"q": {
        "type": "noul", "instructions": "It mentions coffee."}}}
    (_, _, record), = to_records(request)
    assert record["state"] == json.dumps({"msg": "café"}, ensure_ascii=False)


@pytest.mark.parametrize("request_body", [
    {"state": "", "questions": {"q": {"type": "noul", "instructions": "x"}}},
    {"state": "s", "questions": {}},
    {"state": "s", "questions": {"q": {"type": "rank", "instructions": "x"}}},
    {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "only"}}}},
    {"state": "s", "permutations": 2, "questions": {"q": {"type": "noul", "instructions": "x"}}},
    {"state": "s", "extra": 1, "questions": {"q": {"type": "noul", "instructions": "x"}}},
])
def test_invalid_requests_are_rejected(request_body):
    with pytest.raises((ValueError, TypeError)):
        to_records(request_body)


def test_model_id_is_checked_only_when_given():
    request = {"model": "other", "state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}
    to_records(request)
    with pytest.raises(ValueError):
        to_records(request, model_id="instinct-tuned-4b")


def test_answers_have_the_documented_shape():
    assert to_answer("noul", {"probabilities": {"yes": 0.25, "no": 0.75}}) == {"type": "noul", "noul": 0.25}
    score = to_answer("score", {"probabilities": {"0": 0.5, "1": 0.25, "2": 0.25}})
    assert score["score"] == 0.75
    choice = to_answer("choice", {"probabilities": {"a": 0.5, "b": 0.5}})
    assert choice["choice"] == "a" and choice["confidence"] == 0.0
    assert confidence([1.0, 0.0]) == pytest.approx(1.0)


def test_answer_runs_every_question_through_the_model():
    class Stub:
        def decide(self, record):
            n = len(record["criteria"])
            return {"probabilities": {c["id"]: 1 / n for c in record["criteria"]}, "prompt_tokens": 10}

    result = answer(Stub(), body(
        a={"type": "noul", "instructions": "x"},
        b={"type": "choice", "instructions": "y", "criteria": {"k1": "one", "k2": "two"}}))
    assert set(result["answers"]) == {"a", "b"}
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 0}

