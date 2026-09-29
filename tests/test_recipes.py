"""Recipe behaviour that does not need a GPU."""

import pytest

from instinct.recipes import RECIPES, get
from instinct.recipes.base import Encoded


class CharTokenizer:
    """Chat template + one token per character, enough to exercise encode()."""

    all_special_ids = [0]

    def apply_chat_template(self, messages, **kwargs):
        return "<u>" + messages[0]["content"] + "</u><a>"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


RECORD = {
    "id": "x", "group_id": "g", "state": "s", "instructions": "Pick.", "primitive": "choice",
    "criteria": [{"id": "a", "description": "First"}, {"id": "b", "description": "Second"}],
}


def test_registry_names_match_recipes():
    assert RECIPES and all(name == recipe.name for name, recipe in RECIPES.items())
    with pytest.raises(ValueError):
        get("nope")


def test_tuned_4b_encodes_single_token_labels():
    recipe = get("instinct-tuned-4b")
    encoded = recipe.encode(CharTokenizer(), RECORD, ["a", "b"])
    assert encoded.label_token_ids == [ord("A"), ord("B")]
    assert encoded.candidates == ["a", "b"] and encoded.readout_index == -1
    reordered = recipe.encode(CharTokenizer(), RECORD, ["b", "a"])
    assert reordered.candidates == ["b", "a"]


def test_merge_maps_branches_back_and_averages():
    recipe = get("instinct-tuned-4b")
    branches = [Encoded([1], [1, 2], ["a", "b"]), Encoded([1], [1, 2], ["b", "a"])]
    # Branch 2 puts "b" first; identical logits per candidate must give identical output.
    probs = recipe.merge(RECORD, branches, [[1.0, 0.0], [0.0, 1.0]])
    single = recipe.merge(RECORD, branches[:1], [[1.0, 0.0]])
    assert probs == pytest.approx(single)
    assert sum(probs) == pytest.approx(1.0)
