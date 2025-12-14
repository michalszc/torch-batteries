"""Tests for torch_batteries.utils.progress.ProgressMetrics TypedDict."""

from torch_batteries.utils.progress import ProgressMetrics  # noqa: TC001


class TestProgressMetrics:
    """Test cases for ProgressMetrics TypedDict."""

    def test_progress_metrics_with_loss(self) -> None:
        """Test ProgressMetrics with loss value."""
        metrics: ProgressMetrics = {"loss": 0.5}
        assert "loss" in metrics
        assert metrics["loss"] == 0.5

    def test_progress_metrics_empty(self) -> None:
        """Test ProgressMetrics can be empty."""
        metrics: ProgressMetrics = {}
        assert "loss" not in metrics
