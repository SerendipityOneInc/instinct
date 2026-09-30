"""A recipe fixes everything between a request and the candidate probabilities.

Weights alone do not define a decision model: the prompt, the label tokens, the
readout position, the option orders and the temperature all change the output.
Each released model is one Recipe. InstinctModel runs it; the recipe itself
never touches torch.
"""

import math
from dataclasses import dataclass, field

from ..calibration import calibrate

QUESTION_TYPES = ("noul", "choice", "score")
# Question type of a record that does not carry one (e.g. canary records.jsonl).
_TYPE_OF_PRIMITIVE = {"noul": "noul", "choice": "choice", "score_level": "score"}


@dataclass(frozen=True)
class Encoded:
    """One forward pass: prompt token ids and the label token id per candidate.

    ``readout_index`` is the position whose logits are read (default: last).
    ``candidates`` are candidate ids in the order the labels were assigned.
    """

    input_ids: list
    label_token_ids: list
    candidates: list
    readout_index: int = -1


@dataclass(frozen=True)
class Recipe:
    name: str
    hf_repo: str
    revision: str | None
    temperature: float
    readout: str  # "candidate_rows" (final norm + selected LM-head rows) or "full_head"
    max_length: int
    description: str = ""
    extra: dict = field(default_factory=dict)
    # Optional per-question-type temperature; types not listed use ``temperature``.
    temperature_by_type: dict = field(default_factory=dict)

    def __post_init__(self):
        for kind, value in self.temperature_by_type.items():
            if kind not in QUESTION_TYPES:
                raise ValueError(f"temperature_by_type: unknown question type {kind!r}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                raise ValueError(f"temperature_by_type[{kind!r}] must be finite and positive")

    def temperature_for(self, question_type):
        """Serving temperature for ``question_type`` ("noul", "choice" or "score")."""
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"unknown question type {question_type!r}")
        return self.temperature_by_type.get(question_type, self.temperature)

    @staticmethod
    def question_type(record):
        """The SystemOne question type of ``record``.

        ``to_records`` stamps it as ``question_type`` (a score question is a
        ``choice`` primitive); records without the stamp map from the primitive.
        """
        return record.get("question_type") or _TYPE_OF_PRIMITIVE[record["primitive"]]

    def record_temperature(self, record):
        return self.temperature_for(self.question_type(record))

    # --- request side -------------------------------------------------------

    def records(self, body, model_id=None):
        """SystemOne body -> [(question id, type, record)]. Override if needed."""
        from ..systemone import to_records

        return to_records(body, model_id)

    def orders(self, record):
        """Candidate-id orders to score; the first is the order given."""
        return [[c["id"] for c in record["criteria"]]]

    def encode(self, tokenizer, record, order):
        """Return an Encoded for ``record`` with labels assigned in ``order``."""
        raise NotImplementedError

    # --- output side --------------------------------------------------------

    def merge(self, record, encoded_branches, branch_logits):
        """Temperature-softmax each branch, map back to candidate ids, average.

        The temperature is ``record_temperature(record)``: the recipe's
        ``temperature_by_type`` entry for the record's question type, else
        ``temperature``.
        """
        temperature = self.record_temperature(record)
        original = [c["id"] for c in record["criteria"]]
        totals = dict.fromkeys(original, 0.0)
        for encoded, logits in zip(encoded_branches, branch_logits):
            if sorted(encoded.candidates) != sorted(original):
                raise ValueError("branch candidates must permute the record's candidates")
            for candidate, p in zip(encoded.candidates, calibrate(logits, temperature)):
                totals[candidate] += p
        merged = [totals[c] / len(branch_logits) for c in original]
        total = sum(merged)
        return [p / total for p in merged]
