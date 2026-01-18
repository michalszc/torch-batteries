"""Unit tests for ExperimentTrackingCallback."""

from typing import Any

from torch_batteries import EventContext
from torch_batteries.callbacks.experiment_tracking import ExperimentTrackingCallback
from torch_batteries.tracking.base import ExperimentTracker
from torch_batteries.tracking.types import Run


class FakeTracker(ExperimentTracker):
    """A fake `ExperimentTracker` for testing without external dependencies."""

    def __init__(self):
        """Create a fake tracker."""
        self.initialized = False
        self.logged_metrics: list[dict[str, Any]] = []
        self.config_updates: list[dict[str, Any]] = []
        self.summary_data: dict[str, Any] = {}
        self.run: Run | None = None
        self.finished = False
        self.exit_code: int | None = None

    def init(self, run: Run) -> None:
        """Initialize the tracker."""
        self.initialized = True
        self.run = run

    @property
    def is_initialized(self) -> bool:
        """Check if tracker is initialized."""
        return self.initialized

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        """Log metrics."""
        self.logged_metrics.append(
            {"metrics": metrics.copy(), "step": step, "prefix": prefix}
        )

    def update_config(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        self.config_updates.append(config.copy())

    def finish(self, exit_code: int = 0) -> None:
        """Finish the tracker."""
        self.initialized = False
        self.finished = True
        self.exit_code = exit_code

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log summary statistics."""
        self.summary_data.update(summary)


class TestExperimentTrackingCallback:
    """Test suite for ExperimentTrackingCallback."""

    def test_create_with_run(self):
        """Test callback creation with a run."""
        tracker = FakeTracker()
        run = Run(config={"lr": 0.001})
        callback = ExperimentTrackingCallback(
            tracker=tracker, run=run, log_every_n_steps=1
        )

        assert callback.tracker is tracker
        assert callback.run is run
        assert callback.log_every_n_steps == 1
        assert callback.current_epoch == 0
        assert callback.global_step == 0

    def test_create_without_run(self):
        """Test callback creation without a run."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker)

        assert callback.tracker is tracker
        assert callback.run is None
        assert callback.log_every_n_steps == 1

    def test_create_custom_log_frequency(self):
        """Test callback creation with custom log frequency."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker, log_every_n_steps=10)

        assert callback.log_every_n_steps == 10

    def test_on_train_start(self):
        """Test on_train_start initializes tracker."""
        tracker = FakeTracker()
        run = Run(config={"lr": 0.001, "batch_size": 32})
        callback = ExperimentTrackingCallback(tracker=tracker, run=run)

        ctx = EventContext()
        callback.on_train_start(ctx)

        assert tracker.is_initialized
        assert tracker.run == run

    def test_on_train_start_without_run(self):
        """Test on_train_start initializes tracker with default run."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker)

        ctx = EventContext()
        callback.on_train_start(ctx)

        assert tracker.is_initialized
        assert tracker.run is not None

    def test_on_epoch_start(self):
        """Test on_epoch_start updates current epoch."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker)

        ctx = EventContext()
        ctx["epoch"] = 5
        callback.on_epoch_start(ctx)

        assert callback.current_epoch == 5

    def test_on_epoch_start_no_epoch(self):
        """Test on_epoch_start with no epoch in context."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker)

        ctx = EventContext()
        callback.on_epoch_start(ctx)

        assert callback.current_epoch == 0

    def test_on_train_step_end_basic(self):
        """Test on_train_step_end logs metrics."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker, log_every_n_steps=1)

        # Initialize tracker first
        ctx = EventContext()
        callback.on_train_start(ctx)

        # Simulate training step
        ctx["loss"] = 0.5
        ctx["epoch"] = 1
        callback.on_epoch_start(ctx)
        callback.on_train_step_end(ctx)

        assert callback.global_step == 1
        assert len(tracker.logged_metrics) == 1
        logged = tracker.logged_metrics[0]
        assert logged["prefix"] == "train/"
        assert logged["step"] == 1
        assert logged["metrics"]["epoch"] == 1.0
        assert logged["metrics"]["loss"] == 0.5

    def test_on_train_step_end_with_train_metrics(self):
        """Test on_train_step_end logs train_metrics from context."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker, log_every_n_steps=1)

        ctx = EventContext()
        callback.on_train_start(ctx)

        ctx["loss"] = 0.5
        ctx["epoch"] = 1
        callback.on_epoch_start(ctx)
        ctx["train_metrics"] = {"accuracy": 0.95, "f1": 0.92}
        callback.on_train_step_end(ctx)

        assert len(tracker.logged_metrics) == 1
        logged = tracker.logged_metrics[0]
        assert logged["metrics"]["epoch"] == 1.0
        assert logged["metrics"]["loss"] == 0.5
        assert logged["metrics"]["accuracy"] == 0.95
        assert logged["metrics"]["f1"] == 0.92

    def test_on_train_step_end_log_frequency(self):
        """Test on_train_step_end respects log frequency."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker, log_every_n_steps=3)

        ctx = EventContext()
        callback.on_train_start(ctx)
        ctx["loss"] = 0.5

        # First step - not logged (step 1 % 3 != 0)
        callback.on_train_step_end(ctx)
        assert len(tracker.logged_metrics) == 0
        assert callback.global_step == 1

        # Second step - not logged (step 2 % 3 != 0)
        callback.on_train_step_end(ctx)
        assert len(tracker.logged_metrics) == 0
        assert callback.global_step == 2

        # Third step - logged (step 3 % 3 == 0)
        callback.on_train_step_end(ctx)
        assert len(tracker.logged_metrics) == 1
        assert callback.global_step == 3

        # Fourth step - not logged
        callback.on_train_step_end(ctx)
        assert len(tracker.logged_metrics) == 1
        assert callback.global_step == 4

        # Sixth step - logged
        callback.on_train_step_end(ctx)
        callback.on_train_step_end(ctx)
        assert len(tracker.logged_metrics) == 2
        assert callback.global_step == 6

    def test_on_train_step_end_no_loss(self):
        """Test on_train_step_end without loss in context."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker, log_every_n_steps=1)

        ctx = EventContext()
        callback.on_train_start(ctx)

        ctx["epoch"] = 1
        callback.on_epoch_start(ctx)
        callback.on_train_step_end(ctx)

        assert len(tracker.logged_metrics) == 1
        logged = tracker.logged_metrics[0]
        assert "loss" not in logged["metrics"]
        assert logged["metrics"]["epoch"] == 1.0

    def test_on_validation_end(self):
        """Test on_validation_end logs validation metrics."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker)

        ctx = EventContext()
        callback.on_train_start(ctx)

        # Set current epoch
        callback.current_epoch = 2
        callback.global_step = 100

        ctx["val_metrics"] = {"loss": 0.3, "accuracy": 0.97}
        callback.on_validation_end(ctx)

        assert len(tracker.logged_metrics) == 1
        logged = tracker.logged_metrics[0]
        assert logged["prefix"] == "val/"
        assert logged["step"] == 100
        assert logged["metrics"]["epoch"] == 2.0
        assert logged["metrics"]["loss"] == 0.3
        assert logged["metrics"]["accuracy"] == 0.97

    def test_on_validation_end_no_metrics(self):
        """Test on_validation_end with no val_metrics."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker)

        ctx = EventContext()
        callback.on_train_start(ctx)

        callback.current_epoch = 2
        callback.global_step = 100

        callback.on_validation_end(ctx)

        assert len(tracker.logged_metrics) == 1
        logged = tracker.logged_metrics[0]
        assert logged["metrics"]["epoch"] == 2.0
        assert len(logged["metrics"]) == 1  # Only epoch

    def test_on_train_end(self):
        """Test on_train_end finishes tracker."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker)

        ctx = EventContext()
        callback.on_train_start(ctx)

        callback.on_train_end(ctx)

        assert tracker.finished
        assert tracker.exit_code == 0
        assert not tracker.is_initialized

    def test_full_training_lifecycle(self):
        """Test full training lifecycle with callback."""
        tracker = FakeTracker()
        run = Run(config={"lr": 0.001, "epochs": 2})
        callback = ExperimentTrackingCallback(
            tracker=tracker, run=run, log_every_n_steps=1
        )

        # Start training
        ctx = EventContext()
        callback.on_train_start(ctx)
        assert tracker.is_initialized

        # Epoch 1
        ctx["epoch"] = 1
        callback.on_epoch_start(ctx)

        # Training steps
        ctx["loss"] = 0.5
        callback.on_train_step_end(ctx)
        ctx["loss"] = 0.4
        callback.on_train_step_end(ctx)

        # Validation
        ctx["val_metrics"] = {"loss": 0.35, "accuracy": 0.95}
        callback.on_validation_end(ctx)

        # Epoch 2
        ctx["epoch"] = 2
        callback.on_epoch_start(ctx)

        # Training steps
        ctx["loss"] = 0.3
        callback.on_train_step_end(ctx)
        ctx["loss"] = 0.2
        callback.on_train_step_end(ctx)

        # Validation
        ctx["val_metrics"] = {"loss": 0.25, "accuracy": 0.97}
        callback.on_validation_end(ctx)

        # End training
        callback.on_train_end(ctx)

        # Verify
        assert tracker.finished
        assert callback.global_step == 4
        assert callback.current_epoch == 2
        # 4 training steps + 2 validation = 6 log calls
        assert len(tracker.logged_metrics) == 6

        # Check train logs have train/ prefix
        train_logs = [m for m in tracker.logged_metrics if m["prefix"] == "train/"]
        assert len(train_logs) == 4

        # Check val logs have val/ prefix
        val_logs = [m for m in tracker.logged_metrics if m["prefix"] == "val/"]
        assert len(val_logs) == 2

    def test_multiple_epochs_global_step(self):
        """Test that global_step increments correctly across epochs."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker, log_every_n_steps=1)

        ctx = EventContext()
        callback.on_train_start(ctx)

        # Epoch 1 - 3 steps
        ctx["epoch"] = 1
        callback.on_epoch_start(ctx)
        for _ in range(3):
            ctx["loss"] = 0.5
            callback.on_train_step_end(ctx)

        assert callback.global_step == 3

        # Epoch 2 - 3 more steps
        ctx["epoch"] = 2
        callback.on_epoch_start(ctx)
        for _ in range(3):
            ctx["loss"] = 0.4
            callback.on_train_step_end(ctx)

        assert callback.global_step == 6

        # Verify step numbers in logged metrics
        steps = [m["step"] for m in tracker.logged_metrics]
        assert steps == [1, 2, 3, 4, 5, 6]

    def test_train_metrics_not_dict(self):
        """Test that non-dict train_metrics are ignored."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker, log_every_n_steps=1)

        ctx = EventContext()
        callback.on_train_start(ctx)

        ctx["loss"] = 0.5
        ctx["train_metrics"] = "not a dict"
        callback.on_train_step_end(ctx)

        # Should only log loss and epoch, not the string
        logged = tracker.logged_metrics[0]
        assert "not a dict" not in str(logged["metrics"].values())

    def test_val_metrics_not_dict(self):
        """Test that non-dict val_metrics are ignored."""
        tracker = FakeTracker()
        callback = ExperimentTrackingCallback(tracker=tracker)

        ctx = EventContext()
        callback.on_train_start(ctx)

        ctx["val_metrics"] = ["not", "a", "dict"]
        callback.on_validation_end(ctx)

        # Should only log epoch
        logged = tracker.logged_metrics[0]
        assert len(logged["metrics"]) == 1
        assert "epoch" in logged["metrics"]
