"""Unit tests for WandbTracker."""

import builtins
import sys
from unittest.mock import MagicMock, Mock

import pytest

from torch_batteries.tracking.types import Run
from torch_batteries.tracking.wandb import WandbTracker


@pytest.fixture
def mock_wandb(mocker):
    """Mock the wandb module."""
    # Create a mock wandb module
    mock_wandb_module = MagicMock()
    mock_run = Mock()
    mock_run.id = "test-run-123"
    mock_run.url = "https://wandb.ai/test-entity/test-project/runs/test-run-123"
    mock_run.config = Mock()
    mock_run.summary = {}
    mock_wandb_module.init.return_value = mock_run

    # Add it to sys.modules so it can be imported
    mocker.patch.dict(sys.modules, {"wandb": mock_wandb_module})
    return mock_wandb_module


class TestWandbTracker:
    """Test suite for WandbTracker."""

    def test_init_without_entity(self):
        """Test tracker initialization without entity."""
        tracker = WandbTracker(project="test-project")
        assert tracker.project == "test-project"
        assert tracker.entity is None
        assert not tracker.is_initialized

    def test_init_with_entity(self):
        """Test tracker initialization with entity."""
        tracker = WandbTracker(project="test-project", entity="test-entity")
        assert tracker.project == "test-project"
        assert tracker.entity == "test-entity"
        assert not tracker.is_initialized

    def test_init_run_basic(self, mock_wandb):
        """Test initializing a wandb run with basic configuration."""
        tracker = WandbTracker(project="test-project")
        run = Run(config={"lr": 0.001, "batch_size": 32})

        tracker.init(run=run)

        assert tracker.is_initialized
        mock_wandb.init.assert_called_once_with(
            project="test-project",
            entity=None,
            group=None,
            notes=None,
            tags=[],
            job_type=None,
            name=None,
            config={"lr": 0.001, "batch_size": 32},
        )

    def test_init_run_with_all_options(self, mock_wandb):
        """Test initializing a wandb run with all configuration options."""
        tracker = WandbTracker(project="test-project", entity="test-entity")
        run = Run(
            name="test-run",
            config={"lr": 0.001},
            group="experiment-1",
            description="Test experiment",
            tags=["test", "debug"],
            job_type="train",
        )

        tracker.init(run=run)

        assert tracker.is_initialized
        mock_wandb.init.assert_called_once_with(
            project="test-project",
            entity="test-entity",
            group="experiment-1",
            notes="Test experiment",
            tags=["test", "debug"],
            job_type="train",
            name="test-run",
            config={"lr": 0.001},
        )

    def test_init_run_already_initialized(self, mock_wandb):
        """Test that initializing twice raises RuntimeError."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run())

        with pytest.raises(RuntimeError, match="already initialized"):
            tracker.init(run=Run())

    def test_init_run_missing_wandb(self, mocker):
        """Test that missing wandb raises ImportError."""
        # Patch the import to raise ImportError
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "wandb":
                msg = "No module named 'wandb'"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=mock_import)

        tracker = WandbTracker(project="test-project")
        with pytest.raises(ImportError, match="wandb is not installed"):
            tracker.init(run=Run())

    def test_log_metrics_basic(self, mock_wandb):
        """Test logging metrics without step or prefix."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run())

        metrics = {"loss": 0.5, "accuracy": 0.95}
        tracker.log_metrics(metrics)

        tracker.run.log.assert_called_once_with(metrics)

    def test_log_metrics_with_step(self, mock_wandb):
        """Test logging metrics with a step number."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run())

        metrics = {"loss": 0.5, "accuracy": 0.95}
        tracker.log_metrics(metrics, step=100)

        tracker.run.log.assert_called_once_with(metrics, step=100)

    def test_log_metrics_with_prefix(self, mock_wandb):
        """Test logging metrics with a prefix."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run())

        metrics = {"loss": 0.5, "accuracy": 0.95}
        tracker.log_metrics(metrics, prefix="train/")

        expected_metrics = {"train/loss": 0.5, "train/accuracy": 0.95}
        tracker.run.log.assert_called_once_with(expected_metrics)

    def test_log_metrics_with_step_and_prefix(self, mock_wandb):
        """Test logging metrics with both step and prefix."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run())

        metrics = {"loss": 0.5, "accuracy": 0.95}
        tracker.log_metrics(metrics, step=100, prefix="val/")

        expected_metrics = {"val/loss": 0.5, "val/accuracy": 0.95}
        tracker.run.log.assert_called_once_with(expected_metrics, step=100)

    def test_log_metrics_not_initialized(self):
        """Test that logging metrics without initialization raises RuntimeError."""
        tracker = WandbTracker(project="test-project")

        with pytest.raises(RuntimeError, match="not initialized"):
            tracker.log_metrics({"loss": 0.5})

    def test_log_summary(self, mock_wandb):
        """Test logging summary statistics."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run())

        summary = {"best_accuracy": 0.98, "total_epochs": 50}
        tracker.log_summary(summary)

        assert tracker.run.summary["best_accuracy"] == 0.98
        assert tracker.run.summary["total_epochs"] == 50

    def test_log_summary_not_initialized(self):
        """Test that logging summary without initialization raises RuntimeError."""
        tracker = WandbTracker(project="test-project")

        with pytest.raises(RuntimeError, match="not initialized"):
            tracker.log_summary({"best_accuracy": 0.98})

    def test_finish_success(self, mock_wandb):
        """Test finishing a run successfully."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run())

        tracker.finish(exit_code=0)

        tracker.run.finish.assert_called_once_with(exit_code=0)
        assert not tracker.is_initialized

    def test_finish_with_error(self, mock_wandb):
        """Test finishing a run with an error code."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run())

        tracker.finish(exit_code=1)

        tracker.run.finish.assert_called_once_with(exit_code=1)
        assert not tracker.is_initialized

    def test_finish_not_initialized(self):
        """Test that finishing without initialization raises RuntimeError."""
        tracker = WandbTracker(project="test-project")

        with pytest.raises(RuntimeError, match="not initialized"):
            tracker.finish()

    def test_run_id_property(self, mock_wandb):
        """Test run_id property."""
        tracker = WandbTracker(project="test-project")
        assert tracker.run_id is None

        tracker.init(run=Run())
        assert tracker.run_id == "test-run-123"

    def test_run_url_property(self, mock_wandb):
        """Test run_url property."""
        tracker = WandbTracker(project="test-project")
        assert tracker.run_url is None

        tracker.init(run=Run())
        assert (
            tracker.run_url
            == "https://wandb.ai/test-entity/test-project/runs/test-run-123"
        )

    def test_is_initialized_property(self, mock_wandb):
        """Test is_initialized property through lifecycle."""
        tracker = WandbTracker(project="test-project")
        assert not tracker.is_initialized

        tracker.init(run=Run())
        assert tracker.is_initialized

        tracker.finish()
        assert not tracker.is_initialized

    def test_multiple_log_calls(self, mock_wandb):
        """Test multiple consecutive log_metrics calls."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run())

        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.log_metrics({"loss": 0.4}, step=2)
        tracker.log_metrics({"loss": 0.3}, step=3)

        assert tracker.run.log.call_count == 3

    def test_empty_metrics(self, mock_wandb):
        """Test logging empty metrics dictionary."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run())

        tracker.log_metrics({})

        tracker.run.log.assert_called_once_with({})

    def test_empty_config(self, mock_wandb):
        """Test initialization with empty config."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run(config={}))

        assert tracker.is_initialized
        args = mock_wandb.init.call_args[1]
        assert args["config"] == {}

    def test_none_config(self, mock_wandb):
        """Test initialization with None config."""
        tracker = WandbTracker(project="test-project")
        tracker.init(run=Run(config=None))

        assert tracker.is_initialized
        args = mock_wandb.init.call_args[1]
        assert args["config"] is None
