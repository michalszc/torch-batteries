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

    def __init__(
        self,
        tracker: ExperimentTracker,
        run: Run | None = None,
        log_freq: int = 1,
    ) -> None:
        """
        Initialize the experiment tracking callback.

        Args:
            tracker: The experiment tracker instance
            run: Optional run configuration
            log_freq: How often to log metrics (in steps)
        """
        self.tracker = tracker
        self.run = run
        self.log_freq = log_freq

        self._current_epoch = 0
        self._global_step = 0

    @charge(Event.BEFORE_TRAIN)
    def on_train_start(self, ctx: EventContext) -> None:
        """
        Initialize tracker and log configuration.

        Args:
            ctx: Event context
        """
        self.tracker.init(
            run=self.run,
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

        # Log every log_freq steps
        if self._global_step % self.log_freq != 0:
            return

        # Get metrics from context
        metrics = {
            "epoch": self._current_epoch,
        }
        if ctx.get("loss") is not None:
            metrics["loss"] = float(ctx["loss"])

        if ctx.get("train_metrics"):
            metrics.update(ctx["train_metrics"])

        # Log with train/ prefix
        if metrics:
            self.tracker.log_metrics(
                metrics,
                step=self._global_step,
                prefix="train/",
            )

    @charge(Event.AFTER_VALIDATION)
    def on_validation_end(self, ctx: EventContext) -> None:
        """
        Log validation metrics.

        Args:
            ctx: Event context
        """
        assert self.tracker.is_initialized, "Expected tracker to be initialized."

        val_metrics = ctx.get("val_metrics")
        val_metrics["epoch"] = self._current_epoch
        if val_metrics:
            self.tracker.log_metrics(
                val_metrics,
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

        summary = {
            "total_epochs": self._current_epoch,
            "total_steps": self._global_step,
        }
        if ctx.get("train_metrics"):
            summary["train_metrics"] = ctx["train_metrics"]
        if ctx.get("val_metrics"):
            summary["final_val_metrics"] = ctx["val_metrics"]

        self.tracker.log_summary(summary)

        self.tracker.finish()
        logger.info("Experiment tracking finished")
