"""Reflex prompt construction, markdown style (torch-free subset).

Copyright (c) 2026 Kshetrajna Raghavan. MIT License (see LICENSE).
Taken from https://github.com/kshetrajna12/reflex at revision
b77f03bab8e00d7319e64550e062c1eac8111875, src/reflex/prompt.py. Modified as
listed in SOURCE.txt; the prompt wording and option-order logic are unchanged.

Layout of one request (ChatML, which Qwen instruct models are trained on):

    prefix  = <|im_start|>system ... <|im_end|>
              <|im_start|>user
              # Evidence
              <state>

    branch_i = # Criterion
               <instructions>
               # Options
               A. ...
               B. ...
               Respond with only the letter.<|im_end|>
               <|im_start|>assistant
               <think>, blank line, </think>, blank line   (empty think block)

The prefix and every branch are tokenized separately and concatenated.
"""

from __future__ import annotations

import json
import math
import random
import string
from dataclasses import dataclass, field
from typing import Any

Text = Any  # str | dict | list


@dataclass
class NoulCriteria:
    true: Text | None = None
    false: Text | None = None


@dataclass
class NoulQuestion:
    instructions: Text
    criteria: NoulCriteria | None = None


@dataclass
class ChoiceQuestion:
    instructions: Text
    criteria: dict[str, Text | None] = field(default_factory=dict)


@dataclass
class ScoreQuestion:
    instructions: Text
    criteria: list[Text] = field(default_factory=list)  # ordered low -> high


SYSTEM_PROMPT = (
    "You are a System One decision model. You read the State and answer each Question "
    "by choosing exactly one of the listed options. You never explain. You answer with "
    "the single option label only."
)

LETTERS = string.ascii_uppercase  # A..Z


def render_text(x: Text | None) -> str:
    """Render instructions / criteria / state given as str or JSON-ish into prompt text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, indent=2)


@dataclass
class Branch:
    """One isolated question branch: text + which label tokens to read out."""

    qid: str
    kind: str  # "noul" | "choice" | "score"
    text: str
    labels: list[str]  # label strings whose next-token logit we read, in prompt order
    # option keys (choice), level indices (score) or True/False (noul) in prompt order
    keys: list[Any] = field(default_factory=list)


DEFAULT_TEXTS = {
    "system_prompt": SYSTEM_PROMPT,
    "state_heading": "# Evidence",
    "question_heading": "# Criterion",
    "options_heading": "# Options",
    "choice_ask": "Respond with only the letter of the best option.",
    "score_ask": "Respond with only the letter of the level that best matches.",
    "noul_true_default": "The statement is true.",
    "noul_false_default": "The statement is false.",
    "score_level_prefix": "(level {i} of {n})",
}


@dataclass
class PromptFormat:
    """ChatML wrapping with the Qwen3 empty-think prefix, markdown layout."""

    system_prompt: str = SYSTEM_PROMPT

    def t(self, key: str) -> str:
        return DEFAULT_TEXTS[key]

    def prefix(self, state: Text) -> str:
        body = f"{self.t('state_heading')}\n{render_text(state)}\n\n"
        return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{body}"

    def branch(self, body: str) -> str:
        return f"{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _options_block(
    instructions: Text, labelled: list[tuple[str, str]], ask: str, fmt: PromptFormat
) -> str:
    qh = fmt.t("question_heading")
    oh = fmt.t("options_heading")
    lines = [f"{qh}\n{render_text(instructions)}\n", oh]
    for label, desc in labelled:
        lines.append(f"{label}. {desc}" if desc else f"{label}.")
    lines.append(f"\n{ask}\n")
    return "\n".join(lines)


def distinct_orders(keys: list[Any], permutations: int, rng: random.Random) -> list[list[Any]]:
    """The identity order first, then up to `permutations - 1` *distinct* shuffles. With two
    options the second order is always the swap, so binary questions get balanced coverage."""
    orders = [list(keys)]
    seen = {tuple(keys)}

    limit = min(permutations, math.factorial(len(keys))) if len(keys) <= 8 else permutations
    tries = 0
    while len(orders) < limit and tries < 50 * limit:
        tries += 1
        o = list(keys)
        rng.shuffle(o)
        if tuple(o) not in seen:
            seen.add(tuple(o))
            orders.append(o)
    return orders


def build_branches(
    qid: str,
    q: NoulQuestion | ChoiceQuestion | ScoreQuestion,
    fmt: PromptFormat,
    permutations: int = 1,
    rng: random.Random | None = None,
    seed: int = 0,
) -> list[Branch]:
    """Build 1..permutations branches for a question, each with a distinct option order.
    Orders are drawn from a generator seeded by (seed, qid), so one question's orders do
    not depend on which other questions are in the request. Yes/no questions get the
    swapped order as their second branch."""
    rng = rng or random.Random(f"{seed}:{qid}")

    if isinstance(q, NoulQuestion):
        t = render_text(q.criteria.true) if q.criteria else ""
        f = render_text(q.criteria.false) if q.criteria else ""
        yes_text, no_text = (t or fmt.t("noul_true_default")), (f or fmt.t("noul_false_default"))
        out: list[Branch] = []
        for swapped in [False, True] if permutations > 1 else [False]:
            keys = [False, True] if swapped else [True, False]
            pair = [("A", "yes: " + yes_text), ("B", "no: " + no_text)]
            if swapped:
                pair = [("A", "no: " + no_text), ("B", "yes: " + yes_text)]
            body = _options_block(q.instructions, pair, fmt.t("choice_ask"), fmt)
            out.append(Branch(qid, "noul", fmt.branch(body), ["A", "B"], keys))
        return out

    if isinstance(q, ChoiceQuestion):
        keys = list(q.criteria.keys())
        descs = {k: (render_text(v) if v is not None else "") for k, v in q.criteria.items()}
        kind = "choice"
        ask = fmt.t("choice_ask")

        def desc_of(k):
            d = descs[k]
            return f"{k}: {d}" if d else k

    else:  # ScoreQuestion: ordered levels, never reordered semantically but may be permuted
        keys = list(range(len(q.criteria)))
        kind = "score"
        ask = fmt.t("score_ask")
        prefix = fmt.t("score_level_prefix")

        def desc_of(k):
            head = prefix.replace("{i}", str(k)).replace("{n}", str(len(keys) - 1))
            return f"{head} {render_text(q.criteria[k])}"

    out: list[Branch] = []
    for order in distinct_orders(keys, permutations, rng):
        labels = [LETTERS[i] for i in range(len(order))]
        labelled = [(lab, desc_of(k)) for lab, k in zip(labels, order)]
        body = _options_block(q.instructions, labelled, ask, fmt)
        out.append(Branch(qid, kind, fmt.branch(body), labels, order))
    return out
