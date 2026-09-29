"""Deterministic dispatch of paired option orders to two independent engines."""

from collections.abc import Callable
from concurrent.futures import Executor, wait
from typing import Any


def validate_dual_backend_config(
    backend_secondary: str | None, batch_size: int, readout: str
) -> None:
    """Reject secondary backends that cannot preserve the dual-order contract."""
    if not backend_secondary:
        return
    if batch_size != 1:
        raise ValueError("--backend-secondary requires --backend-batch-size 1")
    if readout not in ("native", "native-padded"):
        raise ValueError(
            "--backend-secondary requires the native or native-padded readout"
        )


def score_order_pairs(
    executor: Executor,
    score_chunk: Callable[..., list[list[float]]],
    primary: Any,
    secondary: Any,
    prompts: list[list[int]],
    label_ids: list[list[int]],
    all_ids: list[int],
    padded: bool,
) -> list[list[float]]:
    """Score even/odd order pairs concurrently and preserve their input order."""
    if len(prompts) != len(label_ids) or len(prompts) % 2:
        raise RuntimeError("dual backend requires complete option-order pairs")
    scores = []
    for start in range(0, len(prompts), 2):
        futures = (
            executor.submit(
                score_chunk,
                primary,
                prompts[start : start + 1],
                label_ids[start : start + 1],
                all_ids,
                padded,
            ),
            executor.submit(
                score_chunk,
                secondary,
                prompts[start + 1 : start + 2],
                label_ids[start + 1 : start + 2],
                all_ids,
                padded,
            ),
        )
        # Keep the caller's serialization lock held until both engines are
        # idle, even when one side fails.  Otherwise an orphaned request can
        # overlap the next pair and violate the batch-one numeric contract.
        wait(futures)
        for future in futures:
            scores.extend(future.result())
    return scores
