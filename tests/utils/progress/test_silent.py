"""Tests for torch_batteries.utils.progress.SilentProgress class."""

from unittest.mock import MagicMock, patch

from torch_batteries.utils.progress import Phase, SilentProgress


class TestSilentProgress:
    """Test cases for SilentProgress class."""

    def test_init(self) -> None:
        """Test SilentProgress initialization."""
        progress = SilentProgress(total_epochs=5)
        assert progress is not None

    @patch("builtins.print")
    def test_start_epoch_no_output(self, mock_print: MagicMock) -> None:
        """Test start_epoch produces no output."""
        progress = SilentProgress(total_epochs=5)
        progress.start_epoch(2)
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_start_phase_no_output(self, mock_print: MagicMock) -> None:
        """Test start_phase produces no output."""
        progress = SilentProgress()
        progress.start_phase(Phase.TRAIN, total_batches=10)
        mock_print.assert_not_called()

    def test_update_accumulates_loss(self) -> None:
        """Test that update accumulates loss correctly."""
        progress = SilentProgress()
        progress.start_phase(Phase.TRAIN, total_batches=2)

        progress.update({"loss": 0.5}, 10)
        progress.update({"loss": 0.3}, 20)

        avg_loss = progress.end_phase()
        expected_avg = (0.5 * 10 + 0.3 * 20) / (10 + 20)
        assert abs(avg_loss - expected_avg) < 1e-6

    def test_end_phase_returns_zero_no_samples(self) -> None:
        """Test end_phase returns 0 when no samples processed."""
        progress = SilentProgress()
        progress.start_phase(Phase.TRAIN, total_batches=1)
        avg_loss = progress.end_phase()
        assert avg_loss == 0.0

    @patch("builtins.print")
    def test_end_epoch_no_output(self, mock_print: MagicMock) -> None:
        """Test end_epoch produces no output."""
        progress = SilentProgress()
        progress.end_epoch()
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_end_training_no_output(self, mock_print: MagicMock) -> None:
        """Test end_training produces no output."""
        progress = SilentProgress()
        progress.end_training()
        mock_print.assert_not_called()
