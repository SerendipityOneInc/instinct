"""A recipe fixes everything between a request and the candidate probabilities.

Weights alone do not define a decision model: the prompt, the label tokens, the
readout position, the option orders and the temperature all change the output.
Each released model is one Recipe. InstinctModel runs it; the recipe itself
never touches torch.
"""

from dataclasses import dataclass, field

from ..calibration import calibrate


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
        """Temperature-softmax each branch, map back to candidate ids, average."""
        original = [c["id"] for c in record["criteria"]]
        totals = dict.fromkeys(original, 0.0)
        for encoded, logits in zip(encoded_branches, branch_logits):
            if sorted(encoded.candidates) != sorted(original):
                raise ValueError("branch candidates must permute the record's candidates")
            for candidate, p in zip(encoded.candidates, calibrate(logits, self.temperature)):
                totals[candidate] += p
        merged = [totals[c] / len(branch_logits) for c in original]
        total = sum(merged)
        return [p / total for p in merged]
