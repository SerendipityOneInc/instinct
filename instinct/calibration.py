"""Temperature scaling of candidate logits."""

import math


def calibrate(logits, temperature):
    """Return softmax(logits / temperature) as a list of floats."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("nonfinite or missing logits")
    shifted = [(x - max(logits)) / temperature for x in logits]
    weights = [math.exp(x) for x in shifted]
    total = sum(weights)
    return [x / total for x in weights]
