"""instinct-tuned-4b: Qwen3.5-4B fine-tune, APUS-OpenJev-v1 prompt contract v2."""

from .._vendor.apus_runtime.contracts import (
    LABELS,
    PROMPT_VERSION,
    render_ordered_prompt,
    render_prompt,
    validate_request,
)
from .base import Encoded, Recipe


class Tuned4B(Recipe):
    def encode(self, tokenizer, record, order):
        """Adapted from the APUS-OpenJev-v1 reference runtime (MIT).

        Chat template without thinking; every label must be one non-special
        token that tokenizes identically when appended to the prompt.
        """
        validate_request(record)
        given = [c["id"] for c in record["criteria"]]
        text = render_prompt(record) if order == given else render_ordered_prompt(record, order)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > self.max_length:
            raise ValueError("input exceeds the runtime limit; no truncation is performed")
        tokens = []
        for label in LABELS[: len(order)]:
            token = tokenizer.encode(label, add_special_tokens=False)
            joint = tokenizer.encode(prompt + label, add_special_tokens=False)
            if len(token) != 1 or joint != ids + token:
                raise ValueError("candidate label is not a single token at the answer boundary")
            if token[0] in tokenizer.all_special_ids:
                raise ValueError("candidate label must not be a special token")
            tokens.append(token[0])
        if len(set(tokens)) != len(tokens):
            raise ValueError("candidate token ids must be unique")
        return Encoded(ids, tokens, list(order))


RECIPE = Tuned4B(
    name="instinct-tuned-4b",
    hf_repo="<hf-org>/instinct-tuned-4b",
    revision=None,
    temperature=2.8,
    readout="candidate_rows",
    max_length=8192,
    description="Qwen3.5-4B LoRA fine-tune; prompt " + PROMPT_VERSION + "; one option order; T=2.80",
    extra={"prompt_version": PROMPT_VERSION, "full_depth": 32},
)
