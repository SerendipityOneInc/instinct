"""Project selected native dense LM-head rows before matmul, preserving autograd.

The caller must pass the actual output head, never an unwrapped adapter base_layer.
Unsupported heads raise; there is deliberately no automatic full-head fallback.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import module as module_hooks


class UnsupportedCandidateHead(ValueError):
    """The head's semantics cannot be reproduced by plain selected-row linear."""


def _check_head(head):
    if type(head) is not nn.Linear:
        raise UnsupportedCandidateHead("only exact torch.nn.Linear is supported")
    if (
        head.forward.__func__ is not nn.Linear.forward
        if hasattr(head.forward, "__func__")
        else True
    ):
        raise UnsupportedCandidateHead("overridden forward is unsupported")
    if head._modules or head._buffers or set(head._parameters) != {"weight", "bias"}:
        raise UnsupportedCandidateHead(
            "head contains extra modules, buffers or parameters"
        )
    for name in (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
    ):
        if getattr(head, name, None) or getattr(module_hooks, "_global" + name, None):
            raise UnsupportedCandidateHead("module hooks would be bypassed")
    for value in (head.weight, head.bias):
        if value is None:
            continue
        if (
            type(value) is not nn.Parameter
            or value.is_quantized
            or value.layout != torch.strided
            or not value.is_floating_point()
            or value.device.type == "meta"
        ):
            raise UnsupportedCandidateHead(
                "requires ordinary dense floating-point Parameters"
            )
    if head.weight is None or head.weight.shape != (
        head.out_features,
        head.in_features,
    ):
        raise UnsupportedCandidateHead("invalid dense weight shape")
    if head.bias is not None and (
        head.bias.shape != (head.out_features,)
        or head.bias.dtype != head.weight.dtype
        or head.bias.device != head.weight.device
    ):
        raise UnsupportedCandidateHead(
            "bias shape, dtype or device does not match weight"
        )


def candidate_logits(head, hidden_states, token_ids):
    """Return [..., K] logits in supplied token order, including repeated IDs.

    Dtype/autocast follow F.linear; no explicit precision conversion or detach.
    Shape-dependent floating GEMM rounding may differ from full-vocabulary GEMM.
    This validates indices, not tokenization, semantic labels, or probability mass.
    """
    _check_head(head)
    if type(hidden_states) not in (torch.Tensor, nn.Parameter):
        raise TypeError("hidden_states must be an ordinary Tensor")
    if (
        hidden_states.ndim < 1
        or hidden_states.shape[-1] != head.in_features
        or not hidden_states.is_floating_point()
        or hidden_states.layout != torch.strided
        or hidden_states.device != head.weight.device
    ):
        raise ValueError(
            "hidden_states shape, floating layout or device does not match head"
        )
    if type(token_ids) is torch.Tensor:
        if (
            token_ids.ndim != 1
            or token_ids.dtype != torch.long
            or token_ids.device.type == "meta"
        ):
            raise ValueError("token_ids must be a one-dimensional int64 tensor")
        indices = token_ids.to(device=head.weight.device)
    elif isinstance(token_ids, (list, tuple)):
        if any(type(value) is not int for value in token_ids):
            raise TypeError("token IDs must be integers, not booleans or floats")
        if any(value < 0 or value >= head.out_features for value in token_ids):
            raise ValueError("token ID outside vocabulary")
        indices = torch.tensor(token_ids, dtype=torch.long, device=head.weight.device)
    else:
        raise TypeError("token_ids must be a list, tuple or int64 tensor")
    if not indices.numel():
        raise ValueError("at least one candidate token is required")
    if bool(((indices < 0) | (indices >= head.out_features)).any()):
        raise ValueError("token ID outside vocabulary")
    weight = head.weight.index_select(0, indices)
    bias = None if head.bias is None else head.bias.index_select(0, indices)
    return F.linear(hidden_states, weight, bias)
