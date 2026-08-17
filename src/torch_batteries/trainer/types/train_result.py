"""Training workflow result contract."""

from typing import TypedDict


class TrainResult(TypedDict, total=False):
    """Result from training.

    Attributes:
        train_loss: Average training loss for every completed epoch.
        val_loss: Average validation loss for every completed epoch.
        train_metrics: Named training metric histories.
        val_metrics: Named validation metric histories.
    """

    train_loss: list[float]
    val_loss: list[float]
    train_metrics: dict[str, list[float]]
    val_metrics: dict[str, list[float]]
