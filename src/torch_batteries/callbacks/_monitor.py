"""Shared monitoring-phase compatibility helpers."""

from __future__ import annotations

from typing import Literal

from torch_batteries.utils.logging import get_logger

type MonitorPhase = Literal["train", "val"]

logger = get_logger("Callbacks")


def resolve_monitor_phase(
    phase: MonitorPhase | None,
    *,
    stage: MonitorPhase | None,
    required: bool,
) -> MonitorPhase | None:
    """Resolve the canonical phase and its deprecated stage alias."""
    if phase is not None and stage is not None:
        msg = "phase and deprecated stage cannot both be provided"
        raise TypeError(msg)

    resolved: MonitorPhase | None
    if stage is not None:
        logger.warning("'stage' is deprecated; use 'phase' instead.")
        resolved = stage
    else:
        resolved = phase

    if required and resolved is None:
        msg = "missing required argument: 'phase'"
        raise TypeError(msg)
    if resolved is not None and resolved not in {"train", "val"}:
        msg = "phase must be one of 'train' or 'val'"
        raise ValueError(msg)
    return resolved


def require_metric(metric: str | None) -> str:
    """Raise a stable error when compatibility syntax omitted the metric."""
    if metric is None:
        msg = "missing required argument: 'metric'"
        raise TypeError(msg)
    return metric
