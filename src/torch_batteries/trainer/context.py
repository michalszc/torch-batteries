"""Helpers for building trainer event contexts."""

from torch_batteries.events import EventContext
from torch_batteries.trainer.types import TrainResult


def copy_history_context(results: TrainResult) -> EventContext:
    """Build a copied history context from accumulated train results."""
    return {
        "history_train_loss": list(results["train_loss"]),
        "history_val_loss": list(results["val_loss"]),
        "history_train_metrics": {
            key: list(values) for key, values in results["train_metrics"].items()
        },
        "history_val_metrics": {
            key: list(values) for key, values in results["val_metrics"].items()
        },
    }
