"""Tests for torch_batteries.utils.progress.SimpleProgress class."""

from unittest.mock import MagicMock, patch

from torch_batteries.utils.progress import Phase, SimpleProgress


class TestSimpleProgress:
    """Test cases for SimpleProgress class."""

    def test_init(self) -> None:
        """Test SimpleProgress initialization."""
        progress = SimpleProgress(total_epochs=5)
        assert progress is not None

    @patch("builtins.print")
    def test_start_epoch_no_output(self, mock_print: MagicMock) -> None:
        """Test start_epoch produces no output."""
        progress = SimpleProgress(total_epochs=5)
        progress.start_epoch(2)
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_start_phase_no_output(self, mock_print: MagicMock) -> None:
        """Test start_phase produces no output."""
        progress = SimpleProgress()
        progress.start_phase(Phase.TRAIN, total_batches=10)
        mock_print.assert_not_called()

    def test_update_accumulates_loss(self) -> None:
        """Test that update accumulates loss correctly."""
        progress = SimpleProgress()
        progress.start_phase(Phase.TRAIN, total_batches=2)

        progress.update({"loss": 0.5}, 10)
        progress.update({"loss": 0.3}, 20)

        avg_loss = progress.end_phase()
        expected_avg = (0.5 * 10 + 0.3 * 20) / (10 + 20)
        assert abs(avg_loss - expected_avg) < 1e-6

    @patch("builtins.print")
    @patch("time.time")
    def test_end_epoch_with_val_loss(
        self, mock_time: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test end_epoch prints summary with validation loss."""
        mock_time.side_effect = [0, 10, 15]

        progress = SimpleProgress(total_epochs=3)
        progress.start_epoch(0)
        # Simulate train and validation phases
        progress.start_phase(Phase.TRAIN)
        progress.update({"loss": 0.4}, 32)
        progress.end_phase()
        progress.start_phase(Phase.VALIDATION)
        progress.update({"loss": 0.2}, 16)
        progress.end_phase()
        progress.end_epoch()

        expected = "Epoch 1/3 - Train Loss: 0.4000, Val Loss: 0.2000 (5.00s)"
        mock_print.assert_called_with(expected)

    @patch("builtins.print")
    @patch("time.time")
    def test_end_epoch_without_val_loss(
        self, mock_time: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test end_epoch prints summary without validation loss."""
        mock_time.side_effect = [0, 10, 12]

        progress = SimpleProgress(total_epochs=2)
        progress.start_epoch(1)
        # Simulate only train phase
        progress.start_phase(Phase.TRAIN)
        progress.update({"loss": 0.3}, 32)
        progress.end_phase()
        progress.end_epoch()

        expected = "Epoch 2/2 - Train Loss: 0.3000 (2.00s)"
        mock_print.assert_called_with(expected)

    @patch("builtins.print")
    @patch("time.time")
    def test_end_training_prints_total_time(
        self, mock_time: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test end_training prints total time."""
        mock_time.side_effect = [0, 25]

        progress = SimpleProgress(total_epochs=1)
        progress.end_training()

        mock_print.assert_called_with("Training completed in 25.00s")
