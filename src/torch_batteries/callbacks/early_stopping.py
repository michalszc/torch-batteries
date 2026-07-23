"""Early Stopping Callback for torch-batteries."""

from typing import Any, Literal

from torch import Tensor, nn

from torch_batteries import Battery, Event, EventContext, charge
from torch_batteries.callbacks.base import Callback
from torch_batteries.utils.logging import get_logger

logger = get_logger("EarlyStopping")


class EarlyStopping(Callback):
    """Early stops the training if selected metric doesn't improve after a given patience.

    Args:
        stage: One of 'train' or 'val' to indicate which stage's metric to monitor
        metric: The name of the metric to monitor
        min_delta: Minimum change in the monitored metric to qualify as an improvement
        patience: Number of epochs with no improvement after which training will be stopped
        mode: One of 'min' or 'max'. In 'min' mode, training will stop when the
              monitored metric stops decreasing. In 'max' mode, it will stop
              when the metric stops increasing
        restore_best_weights: If True, restore model weights from the epoch with the
                             best value of the monitored metric
    """  # noqa: E501

    def __init__(  # noqa: PLR0913
        self,
        stage: Literal["train", "val"],
        metric: str,
        *,
        min_delta: float = 0.0,
        patience: int = 5,
        mode: Literal["min", "max"] = "min",
        restore_best_weights: bool = False,
    ) -> None:
        if stage not in {"train", "val"}:
            msg = "stage must be one of 'train' or 'val'"
            raise ValueError(msg)
        if min_delta < 0:
            msg = "min_delta must be greater than or equal to zero"
            raise ValueError(msg)
        if patience < 0:
            msg = "patience must be greater than or equal to zero"
            raise ValueError(msg)

        self._stage = stage
        self._metric = metric
        self._min_delta = min_delta
        self._patience = patience
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

    def state_dict(self) -> dict[str, Any]:
        """Return early-stopping state for resumable checkpoints."""
        state = {
            "best_score": self._best_score,
            "epochs_no_improve": self._epochs_no_improve,
            "best_weights": self._best_weights,
        }
        logger.debug(
            "Serialized early stopping state: best_score=%s, no_improve=%d",
            self._best_score,
            self._epochs_no_improve,
        )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore early-stopping state from a checkpoint."""
        try:
            self._best_score = state_dict["best_score"]
            self._epochs_no_improve = int(state_dict["epochs_no_improve"])
            self._best_weights = state_dict["best_weights"]
        except (KeyError, TypeError, ValueError) as error:
            logger.exception("Invalid early stopping state.")
            msg = "Invalid EarlyStopping checkpoint state."
            raise ValueError(msg) from error
        logger.info(
            "Restored early stopping state: best_score=%s, no_improve=%d",
            self._best_score,
            self._epochs_no_improve,
        )

    @charge(Event.BEFORE_TRAIN)
    def run_on_train_start(self, _: EventContext) -> None:
        """
        Initialize early stopping parameters at the start of training.

        Args:
            _: The event context (not used here).
        """
        self._best_score = None
        self._epochs_no_improve = 0
        self._best_weights = None
        logger.debug("Early stopping state reset for a new training run.")

    @staticmethod
    def _snapshot_weights(model: nn.Module) -> dict[str, Tensor]:
        """Create an immutable CPU snapshot of parameters and buffers."""
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    @charge(Event.AFTER_TRAIN_EPOCH)
    def run_on_epoch_end(self, context: EventContext) -> None:
        """Check early stopping after training epoch ends.

        Args:
            context: Event context containing training metrics.
        """
        if self._stage != "train":
            return

        metrics = context["train_metrics"]
        model = context["model"]
        battery = context["battery"]
        self._check_for_early_stop(metrics, model, battery)

    @charge(Event.AFTER_VALIDATION)
    def run_on_validation_end(self, context: EventContext) -> None:
        """Check early stopping after validation ends.

        Args:
            context: Event context containing validation metrics.
        """
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
        logger.debug(
            "Early stopping comparison: metric=%s, current=%s, best=%s, mode=%s",
            self._metric,
            current_score,
            self._best_score,
            self._mode,
        )
        if self._best_score is None:
            self._best_score = current_score
            if self._restore_best_weights:
                self._best_weights = self._snapshot_weights(model)
            logger.debug(
                "Early stopping baseline recorded: metric=%s, score=%s",
                self._metric,
                current_score,
            )
            return

        if self._monitor_op(current_score, self._best_score):
            self._best_score = current_score
            self._epochs_no_improve = 0
            if self._restore_best_weights:
                self._best_weights = self._snapshot_weights(model)
            logger.debug(
                "Early stopping improvement recorded: metric=%s, score=%s",
                self._metric,
                current_score,
            )
        else:
            self._epochs_no_improve += 1
            logger.debug(
                "Early stopping found no improvement: metric=%s, count=%d, patience=%d",
                self._metric,
                self._epochs_no_improve,
                self._patience,
            )
            if self._epochs_no_improve >= self._patience:
                battery.stop_training = True
                logger.info(
                    "Early stopping applied. No improvement in '%s' for %d epochs.",
                    self._metric,
                    self._patience,
                )

    @charge(Event.AFTER_TRAIN)
    def run_on_train_end(self, context: EventContext) -> None:
        """Restore best model weights after training ends if configured.

        Args:
            context: Event context containing the model.
        """
        if self._restore_best_weights and self._best_weights is not None:
            context["model"].load_state_dict(self._best_weights)
            logger.info("Restored best model weights from early stopping.")
