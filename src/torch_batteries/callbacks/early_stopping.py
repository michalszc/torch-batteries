"""Early Stopping Callback for torch-batteries."""

from typing import Any, Literal

from torch import nn

from torch_batteries import Battery, Event, EventContext, charge
from torch_batteries.utils.logging import get_logger

logger = get_logger("EarlyStopping")


class EarlyStopping:
    """
    Early stops the training if selected metric doesn't improve after a given patience.
    """

    def __init__(  # noqa: PLR0913
        self,
        stage: Literal["train", "val"],
        metric: str,
        *,
        min_delta: float = 0.0,
        patience: int = 5,
        verbose: bool = False,
        mode: Literal["min", "max"] = "min",
        restore_best_weights: bool = False,
    ) -> None:
        """
        Args:
            stage: One of 'train' or 'val' to indicate which stage's metric to monitor.
            metric: The name of the metric to monitor.
            min_delta: Minimum change in the monitored metric to qualify as an
                        improvement.
            patience: Number of epochs with no improvement after which training will be
                        stopped.
            verbose: If True, prints a message when early stopping is triggered.
            mode: One of 'min' or 'max'. In 'min' mode, training will stop when the
                        monitored metric stops decreasing. In 'max' mode, it will stop
                        when the metric stops increasing.
        """
        if stage not in {"train", "val"}:
            msg = "stage must be one of 'train' or 'val'"
            raise ValueError(msg)

        self._stage = stage
        self._metric = metric
        self._min_delta = min_delta
        self._patience = patience
        self._verbose = verbose
        self._restore_best_weights = restore_best_weights
        self._best_weights: dict[str, Any] | None = None

        self._best_score: float | None = None
        self._epochs_no_improve = 0

        if mode not in {"min", "max"}:
            msg = "mode must be one of 'min' or 'max'"
            raise ValueError(msg)
        self._mode = mode
        if self._mode == "min":
            self._monitor_op = lambda current, best: current < best - self._min_delta
        else:
            self._monitor_op = lambda current, best: current > best + self._min_delta

    @property
    def best_score(self) -> float | None:
        """Get the best score observed so far."""
        return self._best_score

    @property
    def best_weights(self) -> dict[str, Any] | None:
        """Get the best model weights observed so far."""
        return self._best_weights

    @charge(Event.BEFORE_TRAIN)
    def run_on_train_start(self, _: EventContext) -> None:
        """
        Initialize early stopping parameters at the start of training.

        Args:
            _: The event context (not used here).
        """
        self._best_score = None
        self._epochs_no_improve = 0

    @charge(Event.AFTER_TRAIN_EPOCH)
    def run_on_epoch_end(self, context: EventContext) -> None:
        if self._stage != "train":
            return

        metrics = context["train_metrics"]
        model = context["model"]
        battery = context["battery"]
        self._check_for_early_stop(metrics, model, battery)

    @charge(Event.AFTER_VALIDATION)
    def run_on_validation_end(self, context: EventContext) -> None:
        if self._stage != "val":
            return

        metrics = context["val_metrics"]
        model = context["model"]
        battery = context["battery"]
        self._check_for_early_stop(metrics, model, battery)

    def _check_for_early_stop(
        self, metrics: dict[str, float], model: nn.Module, battery: Battery
    ) -> None:
        """
        Check if early stopping condition is met and update internal state.

        Args:
            metrics: Dictionary of current metrics.
            model: The model being trained.
        """

        if self._metric not in metrics:
            msg = f"Metric '{self._metric}' not found in training metrics."
            raise ValueError(msg)

        current_score = metrics[self._metric]
        if self._best_score is None:
            self._best_score = current_score
            if self._restore_best_weights:
                self._best_weights = model.state_dict()
            return

        if self._monitor_op(current_score, self._best_score):
            self._best_score = current_score
            self._epochs_no_improve = 0
            if self._restore_best_weights:
                self._best_weights = model.state_dict()
        else:
            self._epochs_no_improve += 1
            if self._epochs_no_improve >= self._patience:
                battery.stop_training = True
                if self._verbose:
                    logger.info(
                        "Early stopping applied. No improvement in '%s' for %d epochs.",
                        self._metric,
                        self._patience,
                    )

    @charge(Event.AFTER_TRAIN)
    def run_on_train_end(self, context: EventContext) -> None:
        if self._restore_best_weights and self._best_weights is not None:
            context["model"].load_state_dict(self._best_weights)
            if self._verbose:
                logger.info("Restored best model weights from early stopping.")
