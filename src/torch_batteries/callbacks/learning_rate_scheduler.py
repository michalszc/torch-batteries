"""Learning-rate scheduler callback."""

from typing import Any, Literal, cast

from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from torch_batteries.callbacks._monitor import (
    MonitorPhase,
    resolve_monitor_phase,
)
from torch_batteries.callbacks.base import Callback
from torch_batteries.events import Event, EventContext, charge
from torch_batteries.utils.logging import get_logger

logger = get_logger("callbacks.learning_rate_scheduler")

SchedulerInterval = Literal["step", "epoch"]
SchedulerPhase = MonitorPhase


class LearningRateScheduler(Callback):
    """Advance a PyTorch learning-rate scheduler during Battery training.

    ``phase`` selects the monitored metrics for ``ReduceLROnPlateau``. The
    deprecated ``stage`` keyword remains a compatibility alias.

    Args:
        scheduler: PyTorch scheduler to advance.
        interval: ``"step"`` after optimizer steps or ``"epoch"`` after epochs.
        phase: Metrics phase for ``ReduceLROnPlateau``.
        metric: Metric name for ``ReduceLROnPlateau``.
        stage: Deprecated alias for ``phase``.
    """

    __slots__ = (
        "_interval",
        "_is_plateau",
        "_metric",
        "_phase",
        "_scheduler",
        "_stepped_epochs",
    )

    def __init__(
        self,
        scheduler: LRScheduler,
        interval: SchedulerInterval = "epoch",
        phase: SchedulerPhase | None = None,
        metric: str | None = None,
        *,
        stage: SchedulerPhase | None = None,
    ) -> None:
        phase = resolve_monitor_phase(phase, stage=stage, required=False)
        if interval not in {"step", "epoch"}:
            logger.error("Unsupported scheduler interval: %s", interval)
            msg = "LearningRateScheduler interval must be 'step' or 'epoch'."
            raise ValueError(msg)
        self._is_plateau = isinstance(scheduler, ReduceLROnPlateau)
        if self._is_plateau:
            if (
                interval != "epoch"
                or phase not in {"train", "validation"}
                or not metric
            ):
                logger.error(
                    "ReduceLROnPlateau requires epoch interval, phase, and metric."
                )
                msg = (
                    "ReduceLROnPlateau requires interval='epoch', "
                    "phase='train' or 'validation', and a metric name."
                )
                raise ValueError(msg)
        elif phase is not None or metric is not None:
            logger.error(
                "phase/metric were configured for a scheduler that ignores metrics."
            )
            msg = "phase and metric are only supported for ReduceLROnPlateau."
            raise ValueError(msg)

        self._scheduler = scheduler
        self._interval = interval
        self._phase = phase
        self._metric = metric
        self._stepped_epochs: set[int] = set()
        logger.info(
            "Learning-rate scheduler configured: "
            "type=%s, interval=%s, phase=%s, metric=%s",
            type(scheduler).__name__,
            interval,
            phase,
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
        """Return current learning rates for diagnostic logging."""
        return [float(group["lr"]) for group in self._scheduler.optimizer.param_groups]

    def _step_without_metric(self) -> None:
        """Advance a scheduler that does not consume a monitored metric."""
        before = self._learning_rates()
        self._scheduler.step()
        logger.debug(
            "Learning-rate scheduler advanced: before=%s, after=%s",
            before,
            self._learning_rates(),
        )

    def _step_plateau(self, context: EventContext, metrics_key: str) -> None:
        """Advance a plateau scheduler using one metrics context mapping."""
        metrics = (
            context.get("train_metrics")
            if metrics_key == "train_metrics"
            else context.get("val_metrics")
        )
        if not isinstance(metrics, dict) or self._metric not in metrics:
            logger.error(
                "Scheduler metric is unavailable: phase=%s, metric=%s",
                self._phase,
                self._metric,
            )
            msg = (
                f"LearningRateScheduler metric '{self._metric}' is unavailable "
                f"for phase '{self._phase}'."
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

    @charge(Event.AFTER_OPTIMIZER_STEP)
    def on_optimizer_step_end(self, context: EventContext) -> None:
        """Advance step schedulers after actual optimizer steps.

        Args:
            context: Optimizer-step event context.
        """
        if (
            self._interval == "step"
            and not self._is_plateau
            and context.get("optimizer_step", True)
        ):
            self._step_without_metric()

    @charge(Event.AFTER_TRAIN_EPOCH)
    def on_train_epoch_end(self, context: EventContext) -> None:
        """Advance ordinary epoch schedulers or train-monitored plateau schedulers.

        Args:
            context: Completed training-epoch context and metrics.
        """
        if self._interval != "epoch":
            return
        epoch = context.get("epoch", 0)
        if self._is_plateau:
            if self._phase != "train":
                return
            self._step_plateau(context, "train_metrics")
        else:
            self._step_without_metric()
        self._stepped_epochs.add(epoch)

    @charge(Event.AFTER_VALIDATION)
    def on_validation_end(self, context: EventContext) -> None:
        """Advance validation-monitored plateau schedulers.

        Args:
            context: Completed validation context and metrics.
        """
        if not self._is_plateau or self._phase != "validation":
            return
        self._step_plateau(context, "val_metrics")
        self._stepped_epochs.add(context.get("epoch", 0))

    @charge(Event.AFTER_TRAIN)
    def on_train_end(self, context: EventContext) -> None:
        """Validate that a requested validation metric was observed.

        Args:
            context: Training-end context containing the final epoch.
        """
        if (
            self._is_plateau
            and self._phase == "validation"
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
            "phase": self._phase,
            "metric": self._metric,
            "scheduler": self._scheduler.state_dict(),
            "stepped_epochs": sorted(self._stepped_epochs),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state after validating its configuration.

        Args:
            state_dict: State returned by :meth:`state_dict`, including legacy
                ``stage`` state.
        """
        has_phase = "phase" in state_dict
        has_stage = "stage" in state_dict
        if has_phase == has_stage:
            logger.error(
                "Learning-rate scheduler checkpoint must contain exactly one "
                "monitoring phase key."
            )
            msg = (
                "LearningRateScheduler checkpoint state must contain exactly one "
                "of 'phase' or legacy 'stage'."
            )
            raise ValueError(msg)
        if has_phase:
            checkpoint_phase = resolve_monitor_phase(
                cast("MonitorPhase | None", state_dict["phase"]),
                stage=None,
                required=False,
            )
        else:
            checkpoint_phase = resolve_monitor_phase(
                None,
                stage=cast("MonitorPhase | None", state_dict["stage"]),
                required=False,
            )
        if (
            state_dict.get("interval") != self._interval
            or checkpoint_phase != self._phase
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
