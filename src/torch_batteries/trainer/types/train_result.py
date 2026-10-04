"""Training workflow result contract."""

from typing import TypedDict


class TrainResult(TypedDict, total=False):
    """Result from training.

    Attributes:
        train_loss: Average training loss for every completed epoch.
        train_metrics: Named training metric histories.
        epochs_completed: Cumulative completed epochs, including resumed history.
        optimizer_steps: Cumulative optimizer steps, including resumed history.
        stopped_early: Whether a stop was requested during this run.
        stop_reason: Reason supplied to ``request_stop``, or ``None``.
    """

    train_loss: list[float]
    train_metrics: dict[str, list[float]]
    epochs_completed: int
    optimizer_steps: int
    stopped_early: bool
    stop_reason: str | None
