"""reference-qwen3.8-27b: frozen (untrained) Qwen3.8-27B behind the jqv prompt and readout.

No weights of our own. The base model is used as released; the decision comes
from the logits of the option letters " A", " B", ... at the last prompt
position (softmax over the candidates, temperature 1.0, uncalibrated), with a
single option order.

Reproduces jqv prompt format version 1 / prompt hash 4f85a0b34776 (readout
"native-v1") byte for byte: the same text, the same separately tokenized
prefix (system + document) and suffix (question + options + answer cue), and the
same SystemOne -> candidate mapping (see ``records`` and ``encode``).
"""

import json

from ..prompt import validate_record
from .base import Encoded, Recipe

PROMPT_FORMAT_VERSION = 1
PROMPT_HASH = "4f85a0b34776"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = (
    "You are a decision model. Read the document, then answer each question by choosing "
    "exactly one option. Reply with the option letter only."
)
ANSWER_CUE = "Answer:"
NOUL_DISPLAY = ["no", "yes"]  # noul options are always shown as "no", "yes"


def prefix_text(state):
    """System message and document; the user turn stays open for the question."""
    doc = f"Document:\n{state}\n\n" if state.strip() else ""
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{doc}"


def suffix_text(question, choices):
    """Question, lettered options and the assistant turn (thinking disabled) up to the answer cue."""
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    body = f"Question:\n{question}\n\nOptions:\n{options}\n\nAnswer with the letter only."
    return f"{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{ANSWER_CUE}"


class Jqv27B(Recipe):
    def records(self, body, model_id=None):
        """Generic records, plus the two things jqv renders differently.

        * structured state is shown as pretty-printed JSON (indent=2), not compact;
        * noul keeps the caller's true/false wording (``noul_wording``); the
          record itself stays canonical yes/no so downstream code is unchanged.
        """
        out = super().records(body, model_id)
        state = body["state"]
        pretty = json.dumps(state, ensure_ascii=False, indent=2) if isinstance(state, (dict, list)) else None
        result = []
        for qid, kind, record in out:
            record = dict(record)
            if pretty is not None:
                record["state"] = pretty
            if kind == "noul":
                wording = body["questions"][qid].get("criteria") or {}
                record["noul_wording"] = {k: wording.get(k) for k in ("true", "false")}
            result.append((qid, kind, record))
        return result

    def orders(self, record):
        if record["primitive"] == "noul":
            return [list(NOUL_DISPLAY)]
        return [[c["id"] for c in record["criteria"]]]

    def _choices(self, record, order):
        validate_record(record)
        by_id = {c["id"]: c["description"] for c in record["criteria"]}
        if sorted(order) != sorted(by_id):
            raise ValueError("order must permute the record's candidates")
        if record["primitive"] == "noul":
            wording = record.get("noul_wording") or {}
            text = {"no": wording.get("false"), "yes": wording.get("true")}
            return [f"{c}: {text[c]}" if text[c] else c for c in order]
        return [f"{c}: {by_id[c]}" for c in order]

    def encode(self, tokenizer, record, order):
        choices = self._choices(record, order)
        prefix = tokenizer.encode(prefix_text(record["state"]), add_special_tokens=False)
        suffix = tokenizer.encode(
            suffix_text(record["instructions"].strip(), choices), add_special_tokens=False
        )
        ids = prefix + suffix  # tokenized separately, as in production
        if len(ids) > self.max_length:
            raise ValueError("input exceeds the runtime limit; no truncation is performed")
        letters = []
        for letter in LETTERS:  # every letter must be a single token, as in production
            token = tokenizer.encode(" " + letter, add_special_tokens=False)
            if len(token) != 1:
                raise ValueError(f"readout token {' ' + letter!r} is not a single token")
            letters.append(token[0])
        if len(set(letters)) != len(letters):
            raise ValueError("choice letters map to duplicate token ids")
        return Encoded(ids, letters[: len(order)], list(order))


RECIPE = Jqv27B(
    name="reference-qwen3.8-27b",
    hf_repo="Qwen/Qwen3.8-27B",
    revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    temperature=1.0,
    readout="full_head",
    max_length=32768,
    description=f"Frozen Qwen3.8-27B; jqv prompt v{PROMPT_FORMAT_VERSION}/{PROMPT_HASH}; one option order; T=1.00",
    extra={"prompt_version": f"jqv.prompt.v{PROMPT_FORMAT_VERSION}+{PROMPT_HASH}", "full_depth": 64},
)
