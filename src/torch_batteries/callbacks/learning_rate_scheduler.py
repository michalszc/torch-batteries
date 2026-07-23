"""Learning-rate scheduler callback."""

from typing import Any, Literal, cast

from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from torch_batteries.callbacks.base import Callback
from torch_batteries.events import Event, EventContext, charge
from torch_batteries.utils.logging import get_logger

logger = get_logger("LearningRateScheduler")

SchedulerInterval = Literal["step", "epoch"]
SchedulerStage = Literal["train", "val"]


class LearningRateScheduler(Callback):
    """Advance a PyTorch learning-rate scheduler during Battery training."""

    __slots__ = (
        "_interval",
        "_is_plateau",
        "_metric",
        "_scheduler",
        "_stage",
        "_stepped_epochs",
    )

    def __init__(
        self,
        scheduler: LRScheduler,
        interval: SchedulerInterval = "epoch",
        stage: SchedulerStage | None = None,
        metric: str | None = None,
    ) -> None:
        if interval not in {"step", "epoch"}:
            logger.error("Unsupported scheduler interval: %s", interval)
            msg = "LearningRateScheduler interval must be 'step' or 'epoch'."
            raise ValueError(msg)
        self._is_plateau = isinstance(scheduler, ReduceLROnPlateau)
        if self._is_plateau:
            if interval != "epoch" or stage not in {"train", "val"} or not metric:
                logger.error(
                    "ReduceLROnPlateau requires epoch interval, stage, and metric."
                )
                msg = (
                    "ReduceLROnPlateau requires interval='epoch', "
                    "stage='train' or 'val', and a metric name."
                )
                raise ValueError(msg)
        elif stage is not None or metric is not None:
            logger.error(
                "stage/metric were configured for a scheduler that ignores metrics."
            )
            msg = "stage and metric are only supported for ReduceLROnPlateau."
            raise ValueError(msg)

        self._scheduler = scheduler
        self._interval = interval
        self._stage = stage
        self._metric = metric
        self._stepped_epochs: set[int] = set()
        logger.info(
            "Learning-rate scheduler configured: "
            "type=%s, interval=%s, stage=%s, metric=%s",
            type(scheduler).__name__,
            interval,
            stage,
            metric,
        )

    @property
    def scheduler(self) -> LRScheduler:
        """Underlying PyTorch scheduler."""
        return self._scheduler

    @property
    def interval(self) -> SchedulerInterval:
        """Scheduler advancement interval."""
        return self._interval

    def _learning_rates(self) -> list[float]:
        return [float(group["lr"]) for group in self._scheduler.optimizer.param_groups]

    def _step_without_metric(self) -> None:
        before = self._learning_rates()
        self._scheduler.step()
        logger.debug(
            "Learning-rate scheduler advanced: before=%s, after=%s",
            before,
            self._learning_rates(),
        )

    def _step_plateau(self, context: EventContext, metrics_key: str) -> None:
        metrics = (
            context.get("train_metrics")
            if metrics_key == "train_metrics"
            else context.get("val_metrics")
        )
        if not isinstance(metrics, dict) or self._metric not in metrics:
            logger.error(
                "Scheduler metric is unavailable: stage=%s, metric=%s",
                self._stage,
                self._metric,
            )
            msg = (
                f"LearningRateScheduler metric '{self._metric}' is unavailable "
                f"for stage '{self._stage}'."
            )
            raise ValueError(msg)
        value = float(metrics[self._metric])
        before = self._learning_rates()
        cast("ReduceLROnPlateau", self._scheduler).step(value)
        logger.debug(
            "Plateau scheduler advanced: metric=%s, value=%s, before=%s, after=%s",
            self._metric,
            value,
            before,
            self._learning_rates(),
        )

    @charge(Event.AFTER_TRAIN_STEP)
    def on_train_step_end(self, context: EventContext) -> None:
        """Advance step schedulers after actual optimizer steps."""
        if (
            self._interval == "step"
            and not self._is_plateau
            and context.get("optimizer_step", True)
        ):
            self._step_without_metric()

    @charge(Event.AFTER_TRAIN_EPOCH)
    def on_train_epoch_end(self, context: EventContext) -> None:
        """Advance ordinary epoch schedulers or train-monitored plateau schedulers."""
        if self._interval != "epoch":
            return
        epoch = context.get("epoch", 0)
        if self._is_plateau:
            if self._stage != "train":
                return
            self._step_plateau(context, "train_metrics")
        else:
            self._step_without_metric()
        self._stepped_epochs.add(epoch)

    @charge(Event.AFTER_VALIDATION)
    def on_validation_end(self, context: EventContext) -> None:
        """Advance validation-monitored plateau schedulers."""
        if not self._is_plateau or self._stage != "val":
            return
        self._step_plateau(context, "val_metrics")
        self._stepped_epochs.add(context.get("epoch", 0))

    @charge(Event.AFTER_TRAIN)
    def on_train_end(self, context: EventContext) -> None:
        """Validate that a requested validation metric was observed."""
        if (
            self._is_plateau
            and self._stage == "val"
            and context.get("epoch", -1) not in self._stepped_epochs
        ):
            logger.error(
                "Validation-monitored scheduler did not observe validation metrics."
            )
            msg = (
                "LearningRateScheduler configured for validation requires "
                "a validation loader."
            )
            raise ValueError(msg)

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler and advancement state."""
        return {
            "interval": self._interval,
            "stage": self._stage,
            "metric": self._metric,
            "scheduler": self._scheduler.state_dict(),
            "stepped_epochs": sorted(self._stepped_epochs),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state after validating its configuration."""
        if (
            state_dict.get("interval") != self._interval
            or state_dict.get("stage") != self._stage
            or state_dict.get("metric") != self._metric
        ):
            logger.error("Learning-rate scheduler checkpoint configuration mismatch.")
            msg = "LearningRateScheduler configuration does not match checkpoint state."
            raise ValueError(msg)
        scheduler_state = state_dict.get("scheduler")
        stepped_epochs = state_dict.get("stepped_epochs")
        if not isinstance(scheduler_state, dict) or not isinstance(
            stepped_epochs, list
        ):
            logger.error("Invalid learning-rate scheduler checkpoint state.")
            msg = "Invalid LearningRateScheduler checkpoint state."
            raise TypeError(msg)
        self._scheduler.load_state_dict(scheduler_state)
        self._stepped_epochs = {int(epoch) for epoch in stepped_epochs}
        logger.info(
            "Learning-rate scheduler state restored: rates=%s",
            self._learning_rates(),
        )
