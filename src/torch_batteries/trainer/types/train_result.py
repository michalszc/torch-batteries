"""Training workflow result contract."""

from typing import TypedDict


class TrainResult(TypedDict, total=False):
    """Result from training.

    Attributes:
        train_loss: Average training loss for every completed epoch.
        val_loss: Deprecated validation loss history retained while ``train()``
            temporarily supports validation. Use ``FitResult`` instead.
        train_metrics: Named training metric histories.
        val_metrics: Deprecated validation metric histories retained while
            ``train()`` temporarily supports validation. Use ``FitResult``
            instead.
    """

    train_loss: list[float]
    # Deprecated: validation through train() is temporary; use FitResult.val_loss.
    val_loss: list[float]
    train_metrics: dict[str, list[float]]
    # Deprecated: validation through train() is temporary; use FitResult.val_metrics.
    val_metrics: dict[str, list[float]]
