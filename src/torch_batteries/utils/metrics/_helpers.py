"""Internal metric normalization helpers."""

from typing import Any

import torch

from torch_batteries.utils.logging import get_logger

logger = get_logger("utils.metrics")


def _tensor_samples(tensor: torch.Tensor) -> int:
    """Return the sample count represented by a metric tensor."""
    return tensor.shape[0] if tensor.ndim > 0 else 1


def _metric_float(name: str, value: Any) -> float:
    """Validate and normalize one computed metric value."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            logger.error("Metric '%s' returned a non-scalar tensor.", name)
            msg = f"Metric '{name}' must return a scalar value."
            raise ValueError(msg)
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        logger.exception("Metric '%s' returned a non-numeric value.", name)
        msg = f"Metric '{name}' must return a numeric value."
        raise TypeError(msg) from error
