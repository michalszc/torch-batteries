"""Combined fitting workflow result contract."""

from typing import TypedDict


class FitResult(TypedDict, total=False):
    """Result from fitting.

    Attributes:
        train_loss: Average training loss for every completed epoch.
        val_loss: Average validation loss for every completed epoch, or an empty
            list when validation data was unavailable.
        train_metrics: Named training metric histories.
        val_metrics: Named validation metric histories, or an empty mapping when
            validation data was unavailable.
    """

    train_loss: list[float]
    val_loss: list[float]
    train_metrics: dict[str, list[float]]
    val_metrics: dict[str, list[float]]
