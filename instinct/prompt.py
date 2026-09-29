"""The instinct-tuned-4b decision prompt and its request/response contract.

The rendered prompt text is part of the model: the weights were trained on
exactly these bytes, so any change to the layout below changes the model's
output. A record looks like::

    {"id": str, "group_id": str, "state": str, "instructions": str,
     "primitive": "choice" | "noul" | "score_level",
     "criteria": [{"id": str, "description": str}, ...]}   # 2..16 candidates

Candidate ``i`` (in the order being scored) is shown to the model as the
letter ``LABELS[i]``; the answer is read from the logits of those letters.
"""

import json
import math

PROMPT_VERSION = "instinct.prompt.v1"
LABELS = tuple("ABCDEFGHIJKLMNOP")
PRIMITIVES = ("choice", "noul", "score_level")
# The fixed candidates of a yes/no question (shown to the model verbatim).
YES_NO = (
    {"id": "yes", "description": "The stated proposition is true."},
    {"id": "no", "description": "The stated proposition is false."},
)


def _is_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_record(record):
    """Raise ValueError/TypeError unless ``record`` is a well-formed request."""
    for field in ("id", "group_id", "state", "instructions"):
        if not _is_text(record.get(field)):
            raise ValueError(f"{field} must be a nonempty string")
    primitive = record.get("primitive")
    if primitive not in PRIMITIVES:
        raise ValueError("unsupported primitive")
    criteria = record.get("criteria")
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= len(LABELS):
        raise ValueError(f"criteria must contain 2..{len(LABELS)} candidates")
    seen = set()
    for candidate in criteria:
        if not isinstance(candidate, dict):
            raise TypeError("candidate must be an object")
        for field in ("id", "description"):
            if not _is_text(candidate.get(field)):
                raise ValueError(f"candidate {field} must be nonempty")
        seen.add(candidate["id"])
    if len(seen) != len(criteria):
        raise ValueError("duplicate candidate ids")
    if primitive != "choice" and criteria != list(YES_NO):
        raise ValueError("noul and score_level require canonical yes/no criteria")


def candidate_ids(record):
    return [candidate["id"] for candidate in record["criteria"]]


def render(record, order=None):
    """Prompt text for ``record`` with candidates lettered in ``order``.

    ``order`` is a permutation of the candidate ids (default: as given).
    """
    validate_record(record)
    given = candidate_ids(record)
    order = given if order is None else list(order)
    if sorted(order) != sorted(given) or len(set(order)) != len(order):
        raise ValueError("candidate order must be a permutation of candidate ids")
    description = {c["id"]: c["description"] for c in record["criteria"]}
    letters = LABELS[: len(order)]
    task = {
        "criteria": [
            {"description": description[cid], "label": letter}
            for letter, cid in zip(letters, order)
        ],
        "instructions": record["instructions"],
        "primitive": record["primitive"],
    }
    return (
        f"Shared state:\n{record['state']}\n\n"
        f"{json.dumps(task, ensure_ascii=False, sort_keys=True)}\n"
        f"Return only the selected letter: {', '.join(letters)}.\nAnswer:"
    )


def format_result(record, probabilities):
    """Candidate probabilities (in the record's order) -> result dict."""
    validate_record(record)
    values = [float(p) for p in probabilities]
    if len(values) != len(record["criteria"]):
        raise ValueError("invalid probabilities")
    if not all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in values):
        raise ValueError("invalid probabilities")
    if abs(sum(values) - 1.0) > 1e-5:
        raise ValueError("probabilities must sum to one")
    distribution = dict(zip(candidate_ids(record), values))
    result = {"type": record["primitive"], "probabilities": distribution}
    if record["primitive"] == "choice":
        result["choice"] = max(distribution, key=distribution.get)
    else:
        result["yes_probability"] = distribution["yes"]
    return result
