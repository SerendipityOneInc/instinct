"""Every models/<dir>/ and reference/<dir>/ directory must be complete and self-consistent."""

import json
from pathlib import Path

import pytest

from instinct.prompt import validate_record
from instinct.recipes import RECIPES
from instinct.systemone import to_records

ROOT = Path(__file__).parents[1]
MODELS = sorted(p for root in ("models", "reference") for p in (ROOT / root).iterdir()
                if p.is_dir() and p.name != "TEMPLATE")
REQUIRED = ("README.md", "model.json", "examples/request.json", "examples/expected.json",
            "canary/reference.jsonl")


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


@pytest.mark.parametrize("model", MODELS, ids=lambda p: p.name)
def test_model_directory(model):
    for name in REQUIRED:
        assert (model / name).is_file(), name
    meta = json.loads((model / "model.json").read_text())
    recipe = RECIPES[meta["name"]]  # every model directory has a registered recipe
    assert meta["hf_repo"] == recipe.hf_repo
    assert meta["temperature"] == recipe.temperature
    assert meta.get("temperature_by_type", {}) == recipe.temperature_by_type
    request = json.loads((model / "examples/request.json").read_text())
    expected = json.loads((model / "examples/expected.json").read_text())
    assert set(expected["answers"]) == {qid for qid, _, _ in to_records(request)}
    if (model / "canary/records.jsonl").is_file():
        # Record-level canaries with reference logits (scripts/parity_check.py).
        records = jsonl(model / "canary/records.jsonl")
        reference = {row["id"]: row for row in jsonl(model / "canary/reference.jsonl")}
        assert records and len(records) == len(reference)
        for record in records:
            validate_record(record)
            assert len(reference[record["id"]]["logits"]) == len(record["criteria"])
    else:
        # SystemOne canaries with reference answers (scripts/compare_endpoint.py).
        requests = jsonl(model / "canary/requests.jsonl")
        reference = jsonl(model / "canary/reference.jsonl")
        assert requests and [row["request"] for row in reference] == requests
        for body, row in zip(requests, reference):
            assert set(row["answers"]) == {qid for qid, _, _ in to_records(body)}


def test_every_recipe_has_a_model_directory():
    names = [json.loads((p / "model.json").read_text())["name"] for p in MODELS]
    assert sorted(RECIPES) == sorted(names) and len(set(names)) == len(names)
