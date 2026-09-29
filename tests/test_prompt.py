"""The prompt text is part of the model contract; it must not drift."""

from instinct._vendor.apus_runtime.contracts import render_prompt


def test_prompt_rendering_is_stable():
    record = {
        "id": "x", "group_id": "g", "state": "Shared text", "instructions": "Pick one.",
        "primitive": "choice",
        "criteria": [{"id": "a", "description": "First"}, {"id": "b", "description": "Second"}],
    }
    assert render_prompt(record) == (
        "Shared state:\nShared text\n\n"
        '{"criteria": [{"description": "First", "label": "A"}, '
        '{"description": "Second", "label": "B"}], '
        '"instructions": "Pick one.", "primitive": "choice"}'
        "\nReturn only the selected letter: A, B.\nAnswer:"
    )
