"""instinct-dual-4b: frozen Qwen3.5-4B, Reflex markdown prompt, two option orders.

The model is untrained Qwen3.5-4B. The prompt construction (system message,
"# Evidence" / "# Criterion" / "# Options" layout, lettered options, empty think
prefix, label tokens, option-order rule) is Reflex's (MIT, Copyright (c) 2026
Kshetrajna Raghavan, https://github.com/kshetrajna12/reflex, revision
b77f03bab8e00d7319e64550e062c1eac8111875, prompt style "markdown"); see
``_vendor/reflex``. Our contribution is the serving and inference optimizations
around that prompt (two concurrent workers, one per option-order branch, and a
native padded readout), not the prompt.

Contract, identical to the production service:

* the shared state prefix and each question branch are tokenized separately and
  concatenated (token ids differ from tokenizing the joined text);
* labels are the bare letters "A", "B", ... (no leading space), one token each;
  noul is presented as a lettered pair ``A. yes: ...`` / ``B. no: ...``;
* exactly two option orders per question: the given order and one distinct
  shuffle drawn from ``random.Random(f"0:{question id}")`` (yes/no and other
  two-option questions use the swapped order); the question id is the key of
  the question in the SystemOne request;
* each branch is softmaxed at temperature 1 over its label logits, mapped back
  to the caller's candidate ids, and the two branches are averaged;
* the next-token logits are read at the last position of the unpadded prompt.
  Production appends one ignored token and reads the penultimate position of the
  padded sequence, which is the same computation.
"""

from .._vendor.reflex.prompt import (
    ChoiceQuestion,
    NoulCriteria,
    NoulQuestion,
    PromptFormat,
    ScoreQuestion,
    build_branches,
)
from .base import Encoded, Recipe

REFLEX_REVISION = "b77f03bab8e00d7319e64550e062c1eac8111875"
PROMPT_VERSION = "reflex-markdown@b77f03b"
PERMUTATIONS = 2
MAX_BRANCH_TOKENS = 4096  # per question branch, as in production
MAX_SCORE_LEVELS = 10  # Reflex limit

_FORMAT = PromptFormat()


def _candidate_id(kind, key):
    if kind == "noul":
        return "yes" if key else "no"
    return str(key)


class ReflexDual4B(Recipe):
    def records(self, body, model_id=None):
        """Generic records, plus the raw question for Reflex's own rendering."""
        records = super().records(body, model_id)
        out = []
        for qid, kind, record in records:
            question = body["questions"][qid]
            if kind == "score" and len(question["criteria"]) > MAX_SCORE_LEVELS:
                raise ValueError(f"{qid}: score supports at most {MAX_SCORE_LEVELS} levels")
            reflex = {"kind": kind, "state": body["state"], "criteria": question.get("criteria")}
            out.append((qid, kind, dict(record, reflex=reflex)))
        return out

    @staticmethod
    def _question(record):
        """(state, Reflex question, kind) for a record; plain records are supported too."""
        reflex = record.get("reflex")
        if reflex is None:
            state = record["state"]
            kind = "noul" if record["primitive"] == "noul" else "choice"
            criteria = (
                None if kind == "noul" else {c["id"]: c["description"] for c in record["criteria"]}
            )
        else:
            state, kind, criteria = reflex["state"], reflex["kind"], reflex["criteria"]
        instructions = record["instructions"]
        if kind == "noul":
            question = NoulQuestion(instructions, NoulCriteria(**criteria) if criteria else None)
        elif kind == "score":
            question = ScoreQuestion(instructions, list(criteria))
        else:
            question = ChoiceQuestion(instructions, dict(criteria))
        return state, question, kind

    def _branches(self, record):
        state, question, kind = self._question(record)
        built = build_branches(record["id"], question, _FORMAT, PERMUTATIONS)
        if len(built) != PERMUTATIONS or len({tuple(b.keys) for b in built}) != PERMUTATIONS:
            raise ValueError(f"{record['id']}: could not build two distinct option orders")
        return state, kind, built

    def orders(self, record):
        _, kind, built = self._branches(record)
        return [[_candidate_id(kind, k) for k in b.keys] for b in built]

    def encode(self, tokenizer, record, order):
        state, kind, built = self._branches(record)
        branch = next(
            (b for b in built if [_candidate_id(kind, k) for k in b.keys] == list(order)), None
        )
        if branch is None:
            raise ValueError("order is not one of the two Reflex option orders for this record")
        prefix_ids = tokenizer.encode(_FORMAT.prefix(state), add_special_tokens=False)
        branch_ids = tokenizer.encode(branch.text, add_special_tokens=False)
        if len(branch_ids) > MAX_BRANCH_TOKENS:
            raise ValueError(f"question branch too long: {len(branch_ids)} > {MAX_BRANCH_TOKENS}")
        ids = prefix_ids + branch_ids
        if len(ids) > self.max_length:
            raise ValueError("input exceeds the runtime limit; no truncation is performed")
        tokens = []
        for label in branch.labels:
            token = tokenizer.encode(label, add_special_tokens=False)
            if len(token) != 1:
                raise ValueError(f"label {label!r} is not a single token")
            tokens.append(token[0])
        if len(set(tokens)) != len(tokens):
            raise ValueError("candidate token ids must be unique")
        return Encoded(ids, tokens, list(order))


RECIPE = ReflexDual4B(
    name="instinct-dual-4b",
    hf_repo="Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    temperature=1.0,
    readout="full_head",
    max_length=262144,
    description="Frozen Qwen3.5-4B; Reflex markdown prompt; two option orders; T=1.00",
    extra={"prompt_version": PROMPT_VERSION, "reflex_revision": REFLEX_REVISION, "full_depth": 32},
)
