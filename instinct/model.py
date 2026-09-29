"""Load a released decision model and score requests with its recipe.

The model never generates text. Each option-order branch is one forward pass;
the logits of the candidate label tokens at the recipe's readout position are
temperature-softmaxed, mapped back to candidate ids and averaged.
"""

import json
import os
from pathlib import Path

from .prompt import format_result
from .recipes import get as get_recipe

SUPPORTED_TRANSFORMERS = "5.16.1"
MULTIMODAL_TOKENS = (
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)


def _check_transformers():
    import transformers

    if transformers.__version__ != SUPPORTED_TRANSFORMERS and not os.environ.get(
        "INSTINCT_ALLOW_UNVERIFIED_TRANSFORMERS"
    ):
        raise RuntimeError(
            f"layer execution is verified for transformers=={SUPPORTED_TRANSFORMERS}, "
            f"found {transformers.__version__}; set "
            "INSTINCT_ALLOW_UNVERIFIED_TRANSFORMERS=1 to run anyway"
        )


def _resolve(name_or_path, revision):
    path = Path(name_or_path)
    if path.is_dir():
        if revision is not None:
            raise ValueError("revision applies to HF repo ids, not to a local weights directory")
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, revision=revision))


class InstinctModel:
    """Single-request, text-only decision runtime. Never truncates input."""

    def __init__(self, model, tokenizer, recipe):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.recipe = recipe
        self.device = self.model.get_input_embeddings().weight.device
        if recipe.readout not in ("candidate_rows", "full_head"):
            raise ValueError(f"unknown readout {recipe.readout!r}")

    @property
    def temperature(self):
        return self.recipe.temperature

    @classmethod
    def from_pretrained(cls, name, *, weights=None, revision=None, device="cuda:0",
                        dtype="bfloat16"):
        """Load a released model by name, e.g. ``"instinct-tuned-4b"``.

        ``weights`` overrides where the weights come from (a local directory or
        an HF repo id); by default the recipe's pinned ``hf_repo``/``revision``.
        """
        import torch
        from transformers import AutoConfig, AutoTokenizer

        _check_transformers()
        recipe = get_recipe(name)
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("supported dtypes: float32, bfloat16")
        directory = _resolve(weights or recipe.hf_repo,
                             revision if weights else (revision or recipe.revision))
        decision = directory / "decision_config.json"
        if decision.exists():
            declared = json.loads(decision.read_text())
            if declared.get("temperature") != recipe.temperature:
                raise ValueError("decision_config.json temperature differs from the recipe")
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        if config.model_type == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration as loader
        elif config.model_type == "qwen3_5_text":
            from transformers import AutoModelForCausalLM as loader
        else:
            raise ValueError("expected a Qwen3.5 text or conditional-generation model")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = loader.from_pretrained(
            directory, local_files_only=True, dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        return cls(model, tokenizer, recipe)

    def encode(self, record, order):
        encoded = self.recipe.encode(self.tokenizer, record, order)
        for name in MULTIMODAL_TOKENS:
            token = getattr(self.model.config, name, None)
            if token is not None and token in encoded.input_ids:
                raise ValueError("multimodal placeholders are unsupported")
        return encoded

    def _logits(self, encoded):
        import torch

        input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=self.device)
        labels = torch.tensor(encoded.label_token_ids, dtype=torch.long, device=self.device)
        if self.recipe.readout == "candidate_rows":
            if encoded.readout_index != -1:
                raise ValueError("candidate_rows readout reads the last position only")
            return self._candidate_rows(input_ids, labels)
        if encoded.readout_index >= 0:
            raise ValueError("readout_index must count from the end (-1 = last position)")
        output = self.model(input_ids=input_ids, use_cache=False, return_dict=True,
                            logits_to_keep=-encoded.readout_index)
        return output.logits[0, 0].index_select(0, labels).float().tolist()

    def _text_model(self):
        """The Qwen3.5 text decoder (embeddings, layers, final norm), without the LM head."""
        inner = self.model.model
        return inner.language_model if self.model.config.model_type == "qwen3_5" else inner

    def _candidate_rows(self, input_ids, labels):
        """Full-depth final-norm hidden state at the last position times the label rows of the LM head."""
        import torch.nn.functional as F

        output = self._text_model()(input_ids=input_ids, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state[:, -1]  # already passed through the final norm
        head = self.model.get_output_embeddings()
        rows = head.weight.index_select(0, labels)
        bias = None if head.bias is None else head.bias.index_select(0, labels)
        return F.linear(hidden, rows, bias)[0].float().tolist()

    def decide(self, record):
        """Score one validated record (id, group_id, state, instructions, primitive, criteria)."""
        import torch

        branches = [self.encode(record, order) for order in self.recipe.orders(record)]
        with torch.inference_mode():
            logits = [self._logits(encoded) for encoded in branches]
        probabilities = self.recipe.merge(record, branches, logits)
        response = format_result(record, probabilities)
        response.update(
            prediction=max(response["probabilities"], key=response["probabilities"].get),
            prompt_tokens=sum(len(e.input_ids) for e in branches),
            temperature=self.recipe.temperature,
        )
        if len(branches) == 1:
            response["logits"] = logits[0]
        else:
            response["branches"] = [
                {"order": e.candidates, "logits": branch} for e, branch in zip(branches, logits)
            ]
        return response
