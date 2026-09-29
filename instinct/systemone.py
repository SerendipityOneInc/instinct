"""SystemOne request format: several questions about one shared state.

Request::

    {"state": str | object | array,
     "questions": {qid: {"type": "noul" | "choice" | "score",
                         "instructions": str,
                         "criteria": ...}},
     "model": optional str, "permutations": optional 1}

``noul`` criteria are optional ``{"true": str, "false": str}`` wording; the model
always sees the canonical yes/no candidates. ``choice`` criteria map option keys
to descriptions (2..16). ``score`` criteria are an ordered list of 2..16 level
descriptions, scored as choices "0".."n-1".
"""

import copy
import json
import math

from ._vendor.apus_runtime.contracts import BINARY_CRITERIA, LABELS, validate_request

REQUEST_FIELDS = {"model", "state", "questions", "permutations"}
QUESTION_FIELDS = {"type", "instructions", "criteria"}


def _state(value):
    if isinstance(value, (dict, list)) and value:
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("state must be a nonempty string, object, or array")


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def to_records(body, model_id=None):
    """Validate a request and return [(question id, type, record), ...]."""
    if not isinstance(body, dict):
        raise ValueError("request must be an object")
    extra = set(body) - REQUEST_FIELDS
    if extra:
        raise ValueError(f"unsupported request fields: {sorted(extra)}")
    if model_id is not None and body.get("model") not in (None, model_id):
        raise ValueError(f"model {body.get('model')!r} is not served; expected {model_id!r}")
    if body.get("permutations") not in (None, 1):
        raise ValueError("only one option order (permutations=1) is supported")
    state = _state(body.get("state"))
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    records = []
    for qid, question in questions.items():
        _nonempty(qid, "question id")
        if not isinstance(question, dict):
            raise ValueError(f"{qid}: question must be an object")
        kind = question.get("type")
        if kind not in ("noul", "choice", "score"):
            raise ValueError(f"{qid}: unsupported question type {kind!r}")
        extra = set(question) - QUESTION_FIELDS
        if extra:
            raise ValueError(f"{qid}: unsupported question fields: {sorted(extra)}")
        instructions = _nonempty(question.get("instructions"), f"{qid}: instructions")
        criteria = question.get("criteria")
        if kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict)
                or set(criteria) - {"true", "false"}
                or any(v is not None and not isinstance(v, str) for v in criteria.values())
            ):
                raise ValueError(f"{qid}: noul criteria must map true/false to descriptions")
            candidates = copy.deepcopy(BINARY_CRITERIA)
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= len(LABELS):
                raise ValueError(f"{qid}: score needs a list of 2..{len(LABELS)} level descriptions")
            candidates = [
                {"id": str(level), "description": _nonempty(text, f"{qid}: level {level}")}
                for level, text in enumerate(criteria)
            ]
        else:
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= len(LABELS):
                raise ValueError(f"{qid}: choice needs 2..{len(LABELS)} options")
            candidates = [
                {"id": _nonempty(key, f"{qid}: option key"),
                 "description": _nonempty(text, f"{qid}: option {key!r}")}
                for key, text in criteria.items()
            ]
        record = dict(
            id=qid, group_id="systemone", state=state, instructions=instructions,
            primitive="noul" if kind == "noul" else "choice", criteria=candidates,
        )
        validate_request(record)
        records.append((qid, kind, record))
    return records


def confidence(values):
    """1 - normalized entropy, clipped to [0, 1]."""
    if len(values) <= 1:
        return 1.0
    clipped = [min(max(p, 1e-12), 1.0) for p in values]
    entropy = -sum(p * math.log(p) for p in clipped)
    return float(max(0.0, min(1.0, 1.0 - entropy / math.log(len(values)))))


def to_answer(kind, result):
    probabilities = result["probabilities"]
    if kind == "noul":
        return {"type": "noul", "noul": round(probabilities["yes"], 6)}
    rounded = {str(k): round(v, 6) for k, v in probabilities.items()}
    if kind == "score":
        expected = sum(int(k) * float(v) for k, v in probabilities.items())
        return {"type": "score", "probabilities": rounded, "score": round(expected, 6)}
    return {
        "type": "choice",
        "choice": max(rounded, key=rounded.get),
        "probabilities": rounded,
        "confidence": round(confidence(list(rounded.values())), 6),
    }


def answer(model, body, model_id=None):
    """Run every question of a SystemOne request through ``model.decide``."""
    recipe = getattr(model, "recipe", None)
    records = recipe.records(body, model_id) if recipe else to_records(body, model_id)
    answers, input_tokens = {}, 0
    for qid, kind, record in records:
        result = model.decide(record)
        answers[qid] = to_answer(kind, result)
        input_tokens += result["prompt_tokens"]
    return {"answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
