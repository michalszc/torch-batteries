"""Experiment tracking callback for automatic logging."""

from typing import Any

from torch_batteries import Event, EventContext, charge
from torch_batteries.tracking.base import ExperimentTracker
from torch_batteries.tracking.types import Run
from torch_batteries.utils.logging import get_logger

logger = get_logger("experiment_tracking")


class ExperimentTrackingCallback:
    """
    Callback for automatic experiment tracking during training.

    Integrates an `ExperimentTracker` with `Battery` training loop,
    automatically logging configuration, metrics, and summary.

    This callback hooks into the event system to log:
    - Configuration at training start
    - Training metrics after each step
    - Validation metrics after validation
    - Summary statistics at training end

    Example:
    ```python
    from torch_batteries.tracking import WandbTracker, Run

    # Create tracker and configure run
    tracker = WandbTracker(project="your-wandb-project")
    run = Run(config={"lr": 0.001, "patience": 5})

    # Create callback
    callback = ExperimentTrackingCallback(
        tracker=tracker,
        run=run,
    )

    # Use with Battery
    battery = Battery(model, optimizer=optimizer, callbacks=[callback])
    battery.train(train_loader, val_loader, epochs=10)
    ```
    """

    __slots__ = (
        "_current_epoch",
        "_global_step",
        "_log_every_n_steps",
        "_run",
        "_tracker",
    )

    def __init__(
        self,
        tracker: ExperimentTracker,
        run: Run | None = None,
        log_every_n_steps: int = 1,
    ) -> None:
        """
        Initialize the experiment tracking callback.

        Args:
            tracker: The experiment tracker instance
            run: Optional run configuration
            log_freq: How often to log metrics (in steps)
        """
        self._tracker = tracker
        self._run = run
        self._log_every_n_steps = log_every_n_steps

        self._current_epoch = 0
        self._global_step = 0

    @property
    def tracker(self) -> ExperimentTracker:
        """Get the experiment tracker."""
        return self._tracker

    @property
    def run(self) -> Run | None:
        """Get the run configuration."""
        return self._run

    @property
    def global_step(self) -> int:
        """Current global step."""
        return self._global_step

    @property
    def current_epoch(self) -> int:
        """Current epoch."""
        return self._current_epoch

    @property
    def log_every_n_steps(self) -> int:
        """How often to log metrics (in steps)."""
        return self._log_every_n_steps

    @charge(Event.BEFORE_TRAIN)
    def on_train_start(self, _: EventContext) -> None:
        """
        Initialize tracker and log configuration.

        Args:
            ctx: Event context
        """
        self.tracker.init(
            run=self.run if self.run is not None else Run(),
        )

        logger.info("Experiment tracking started")

    @charge(Event.BEFORE_TRAIN_EPOCH)
    def on_epoch_start(self, ctx: EventContext) -> None:
        """
        Update current epoch.

        Args:
            ctx: Event context
        """
        self._current_epoch = ctx.get("epoch", 0)

    @charge(Event.AFTER_TRAIN_STEP)
    def on_train_step_end(self, ctx: EventContext) -> None:
        """
        Log training metrics after each step.

        Args:
            ctx: Event context
        """
        assert self.tracker.is_initialized, "Expected tracker to be initialized."

        self._global_step += 1

        if self._global_step % self.log_every_n_steps != 0:
            return

        metrics: dict[str, float] = {
            "epoch": float(self._current_epoch),
        }
        if ctx.get("loss") is not None:
            metrics["loss"] = float(ctx["loss"])

        if ctx.get("train_metrics"):
            train_metrics = ctx["train_metrics"]
            if isinstance(train_metrics, dict):
                metrics.update({k: float(v) for k, v in train_metrics.items()})

        self.tracker.log_metrics(
            metrics,
            step=self._global_step,
            prefix="train/",
        )

    @charge(Event.AFTER_VALIDATION_EPOCH)
    def on_validation_epoch_end(self, ctx: EventContext) -> None:
        """
        Log validation metrics.

        Args:
            ctx: Event context
        """
        assert self.tracker.is_initialized, "Expected tracker to be initialized."

        metrics: dict[str, Any] = {}
        metrics["epoch"] = float(self._current_epoch)
        val_metrics = ctx.get("val_metrics")
        if val_metrics and isinstance(val_metrics, dict):
            metrics.update({k: float(v) for k, v in val_metrics.items()})
        self.tracker.log_metrics(
            metrics,
            step=self._global_step,
            prefix="val/",
        )

    @charge(Event.AFTER_TRAIN)
    def on_train_end(self, ctx: EventContext) -> None:
        """
        Log summary and finish tracking.

        Args:
            ctx: Event context
        """
        assert self.tracker.is_initialized, "Expected tracker to be initialized."

        summary: dict[str, Any] = {
            "total_epochs": self._current_epoch,
            "total_steps": self._global_step,
        }
        if ctx.get("train_metrics"):
            summary["train_metrics"] = ctx["train_metrics"]
        if ctx.get("val_metrics"):
            summary["val_metrics"] = ctx["val_metrics"]

        self.tracker.log_summary(summary)

        model = ctx.get("model")
        if model is not None:
            self.tracker.log_model(
                model,
                name="model",
                metadata={
                    "epoch": self._current_epoch,
                    "global_step": self._global_step,
                },
            )

        self.tracker.finish()
        logger.info("Experiment tracking finished")
