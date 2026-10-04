"""Shared monitoring-phase validation."""

from __future__ import annotations

from typing import Literal

from torch_batteries.utils.logging import get_logger

type MonitorPhase = Literal["train", "validation"]

logger = get_logger("callbacks._monitor")


def resolve_monitor_phase(
    phase: MonitorPhase | None,
    *,
    required: bool,
) -> MonitorPhase | None:
    """Validate a monitoring phase.

    Args:
        phase: Requested monitoring phase.
        required: Whether omitting the phase is invalid.
    """
    if required and phase is None:
        msg = "missing required argument: 'phase'"
        raise TypeError(msg)
    if phase is not None and phase not in {"train", "validation"}:
        logger.error("Invalid monitoring phase: %s", phase)
        msg = "phase must be one of 'train' or 'validation'"
        raise ValueError(msg)
    return phase


def require_metric(metric: str | None) -> str:
    """Require a monitored metric name.

    Args:
        metric: Optional metric name supplied by a callback constructor.
    """
    if metric is None:
        msg = "missing required argument: 'metric'"
        raise TypeError(msg)
    return metric
