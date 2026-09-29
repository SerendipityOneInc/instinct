"""Actual Q1 no-cache layer-prefix execution for native Qwen3.5 decisions.

Mirrors the mask/position preparation of Transformers Qwen3_5TextModel (5.16.1).
This is a version-audited reference, not a generic model or cache implementation.
"""

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
)

from .candidate_projection import (
    UnsupportedCandidateHead,
    candidate_logits,
)


@dataclass(frozen=True)
class DepthContinuation:
    hidden: Tensor  # complete sequence residual, BEFORE final norm
    position_ids: Tensor
    position_embeddings: tuple[Tensor, Tensor]
    masks: dict[str, Tensor | None]
    depth: int
    owner: object
    model_signature: tuple


@dataclass
class DepthDecision:
    depth: int
    logits: Tensor  # [1,C]
    projection_mode: str


class QwenEarlyExit(nn.Module):
    """Begin once, stop at a real depth, optionally continue without replay.

    Q1 means one unpadded complete input sequence. No cache or token generation.
    Continuations are ephemeral: do not mutate parameters/train-mode between
    begin/advance/readout, or persist them across optimizer steps.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        if self.base.config.model_type not in {"qwen3_5", "qwen3_5_text"}:
            raise ValueError("only Qwen3.5 text/conditional models are supported")
        if len(self.backbone.layers) != self.backbone.config.num_hidden_layers:
            raise ValueError("layer count/config mismatch")
        if not set(self.backbone.config.layer_types) <= {
            "linear_attention",
            "full_attention",
        }:
            raise ValueError("unsupported hybrid layer type")
        self._owner = object()

    @property
    def base(self):
        return (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )

    @property
    def backbone(self):
        return (
            self.base.model.language_model
            if self.base.config.model_type == "qwen3_5"
            else self.base.model
        )

    @property
    def full_depth(self):
        return len(self.backbone.layers)

    def _signature(self):
        # Reference guard: optimizer updates and mode/device changes invalidate
        # all outstanding continuations. No .data mutation is supported.
        return (
            tuple((id(module), module.training) for module in self.model.modules()),
            tuple(
                (id(parameter), parameter._version, parameter.device, parameter.dtype)
                for parameter in self.model.parameters()
            ),
        )

    def begin(self, input_ids: Tensor) -> DepthContinuation:
        if (
            input_ids.dtype != torch.long
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or input_ids.shape[1] < 1
        ):
            raise ValueError("Q1 requires nonempty int64 input_ids [1,S], no padding")
        for name in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            token = getattr(self.base.config, name, None)
            if token is not None and (input_ids == token).any():
                raise ValueError("multimodal placeholders are unsupported")
        embedding = self.base.get_input_embeddings()
        if ((input_ids < 0) | (input_ids >= embedding.weight.shape[0])).any():
            raise ValueError("input token outside vocabulary")
        hidden = embedding(input_ids)
        positions = (
            torch.arange(input_ids.shape[1], device=hidden.device)
            .view(1, 1, -1)
            .expand(4, 1, -1)
        )
        text_positions = positions[0]
        kwargs = {
            "config": self.backbone.config,
            "inputs_embeds": hidden,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": text_positions,
        }
        masks = {
            "full_attention": create_causal_mask(**kwargs),
            "linear_attention": create_recurrent_attention_mask(**kwargs),
        }
        rotary = self.backbone.rotary_emb(hidden, positions[1:])
        return DepthContinuation(
            hidden, text_positions, rotary, masks, 0, self._owner, self._signature()
        )

    def _check_state(self, state):
        if state.owner is not self._owner:
            raise ValueError("continuation belongs to a different executor")
        if state.model_signature != self._signature():
            raise ValueError(
                "stale continuation: model parameters or training mode changed"
            )

    def advance(self, state: DepthContinuation, target_depth: int) -> DepthContinuation:
        self._check_state(state)
        if (
            type(target_depth) is not int
            or not state.depth < target_depth <= self.full_depth
        ):
            raise ValueError("target depth must advance within the model")
        hidden = state.hidden
        for index in range(state.depth, target_depth):
            hidden = self.backbone.layers[index](
                hidden,
                position_embeddings=state.position_embeddings,
                attention_mask=state.masks[self.backbone.config.layer_types[index]],
                position_ids=state.position_ids,
                past_key_values=None,
                use_cache=False,
            )
        return replace(state, hidden=hidden, depth=target_depth)

    def readout(
        self, state: DepthContinuation, candidate_token_ids: Tensor
    ) -> DepthDecision:
        self._check_state(state)
        if state.depth < 1:
            raise ValueError("readout requires at least one executed layer")
        head = self.base.get_output_embeddings()
        if (
            candidate_token_ids.dtype != torch.long
            or candidate_token_ids.ndim != 1
            or candidate_token_ids.numel() < 2
        ):
            raise ValueError("require at least two int64 candidate tokens [C]")
        if candidate_token_ids.device != state.hidden.device:
            raise ValueError("candidate IDs must share hidden device")
        if candidate_token_ids.unique().numel() != candidate_token_ids.numel():
            raise ValueError("duplicate candidate token")
        if (
            (candidate_token_ids < 0) | (candidate_token_ids >= head.weight.shape[0])
        ).any():
            raise ValueError("candidate token outside vocabulary")
        # Never replace the residual used by continuation with normalized hidden.
        hidden = self.backbone.norm(state.hidden[:, -1])
        try:
            logits = candidate_logits(head, hidden, candidate_token_ids)
            mode = "candidate_rows"
        except UnsupportedCandidateHead:
            # Preserve adapters/hooks/parametrizations by executing the real head.
            logits = head(hidden).index_select(-1, candidate_token_ids)
            mode = "full_head_fallback"
        return DepthDecision(state.depth, logits, mode)

    def forward(
        self, input_ids: Tensor, candidate_token_ids: Tensor, *, depths: tuple[int, ...]
    ):
        if (
            not depths
            or any(type(d) is not int or not 1 <= d <= self.full_depth for d in depths)
            or list(depths) != sorted(set(depths))
        ):
            raise ValueError("depths must be strictly increasing valid layer counts")
        state = self.begin(input_ids)
        decisions = []
        for depth in depths:
            state = self.advance(state, depth)
            decisions.append(self.readout(state, candidate_token_ids))
        return tuple(decisions)
