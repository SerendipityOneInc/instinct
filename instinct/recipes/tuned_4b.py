"""instinct-tuned-4b: Qwen3.5-4B fine-tune, prompt ``instinct.prompt.v1``."""

from ..prompt import LABELS, PROMPT_VERSION, render, validate_record
from .base import Encoded, Recipe


class Tuned4B(Recipe):
    def encode(self, tokenizer, record, order):
        """Chat-templated prompt (thinking disabled) plus one label token per candidate.

        Each letter must be a single, non-special token that the tokenizer keeps
        intact when it directly follows the prompt, so the logit read at the last
        prompt position is the logit of that letter.
        """
        validate_record(record)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": render(record, order)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > self.max_length:
            raise ValueError("input exceeds the runtime limit; no truncation is performed")
        special = set(tokenizer.all_special_ids)
        label_ids = []
        for letter in LABELS[: len(order)]:
            piece = tokenizer.encode(letter, add_special_tokens=False)
            if len(piece) != 1 or tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + piece:
                raise ValueError("candidate label is not a single token at the answer boundary")
            if piece[0] in special:
                raise ValueError("candidate label must not be a special token")
            label_ids.append(piece[0])
        if len(set(label_ids)) != len(label_ids):
            raise ValueError("candidate token ids must be unique")
        return Encoded(ids, label_ids, list(order))


RECIPE = Tuned4B(
    name="instinct-tuned-4b",
    hf_repo="srpone/instinct-tuned-4b",
    revision="fc68951b968589aa4e316a1b0bb2543c4c64886f",
    temperature=2.8,
    readout="candidate_rows",
    max_length=8192,
    description="Qwen3.5-4B LoRA fine-tune; prompt " + PROMPT_VERSION + "; one option order; T=2.80 (noul 0.50)",
    extra={"prompt_version": PROMPT_VERSION, "full_depth": 32},
    temperature_by_type={"noul": 0.5},
)
