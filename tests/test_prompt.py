"""The prompt text is part of the model contract; it must not drift."""

from instinct.prompt import render


def test_prompt_rendering_is_stable():
    record = {
        "id": "x", "group_id": "g", "state": "Shared text", "instructions": "Pick one.",
        "primitive": "choice",
        "criteria": [{"id": "a", "description": "First"}, {"id": "b", "description": "Second"}],
    }
    assert render(record) == (
        "Shared state:\nShared text\n\n"
        '{"criteria": [{"description": "First", "label": "A"}, '
        '{"description": "Second", "label": "B"}], '
        '"instructions": "Pick one.", "primitive": "choice"}'
        "\nReturn only the selected letter: A, B.\nAnswer:"
    )


def test_reordered_prompt_letters_follow_the_order():
    record = {
        "id": "x", "group_id": "g", "state": "s", "instructions": "Pick.",
        "primitive": "choice",
        "criteria": [{"id": "a", "description": "First"}, {"id": "b", "description": "Second"}],
    }
    text = render(record, ["b", "a"])
    assert '{"description": "Second", "label": "A"}, {"description": "First", "label": "B"}' in text
