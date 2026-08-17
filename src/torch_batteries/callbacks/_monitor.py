"""Shared monitoring-phase compatibility helpers."""

from __future__ import annotations

import warnings
from typing import Literal

from torch_batteries.utils.logging import get_logger

type MonitorPhase = Literal["train", "validation", "val"]
type CanonicalMonitorPhase = Literal["train", "validation"]

logger = get_logger("callbacks._monitor")


def resolve_monitor_phase(
    phase: MonitorPhase | None,
    *,
    stage: MonitorPhase | None,
    required: bool,
) -> CanonicalMonitorPhase | None:
    """Resolve the canonical phase and its deprecated stage alias."""
    if phase is not None and stage is not None:
        msg = "phase and deprecated stage cannot both be provided"
        raise TypeError(msg)

    resolved: MonitorPhase | None
    if stage is not None:
        logger.warning("'stage' is deprecated; use 'phase' instead.")
        warnings.warn(
            "'stage' is deprecated; use 'phase' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        resolved = stage
    else:
        resolved = phase

    if required and resolved is None:
        msg = "missing required argument: 'phase'"
        raise TypeError(msg)
    if resolved is not None and resolved not in {"train", "validation", "val"}:
        msg = "phase must be one of 'train' or 'validation'"
        raise ValueError(msg)
    if resolved == "val":
        logger.warning("phase='val' is deprecated; use phase='validation' instead.")
        warnings.warn(
            "phase='val' is deprecated; use phase='validation' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        resolved = "validation"
    return resolved


def require_metric(metric: str | None) -> str:
    """Raise a stable error when compatibility syntax omitted the metric."""
    if metric is None:
        msg = "missing required argument: 'metric'"
        raise TypeError(msg)
    return metric
