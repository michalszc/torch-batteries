"""Standalone validation workflow result contract."""

from typing import TypedDict


class ValidationResult(TypedDict, total=False):
    """Result from one standalone validation pass.

    Attributes:
        val_loss: Average validation loss.
        val_metrics: Named validation metrics when any are produced.
    """

    val_loss: float
    val_metrics: dict[str, float]
