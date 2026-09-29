"""Instinct: a single-forward-pass decision model runtime."""

from .calibration import calibrate
from .systemone import answer, to_records

__all__ = ["InstinctModel", "answer", "calibrate", "to_records"]
__version__ = "0.1.0"


def __getattr__(name):
    # Import torch/transformers lazily so request validation works CPU-only.
    if name == "InstinctModel":
        from .model import InstinctModel

        return InstinctModel
    raise AttributeError(name)
