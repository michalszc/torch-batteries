"""Tests for torch_batteries.utils.progress module."""

from unittest.mock import MagicMock, patch

import pytest

from torch_batteries.utils.progress import Phase, Progress, ProgressMetrics


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


class TestProgress:
    """Test cases for Progress class."""

    def test_init_default_values(self) -> None:
        """Test Progress initialization with default values."""
        progress = Progress()
        assert progress is not None

    def test_init_custom_values(self) -> None:
        """Test Progress initialization with custom values."""
        progress = Progress(verbose=2, total_epochs=10)
        assert progress is not None

    def test_init_invalid_verbose(self) -> None:
        """Test Progress initialization with invalid verbose level."""
        with pytest.raises(ValueError, match="Invalid verbose level"):
            Progress(verbose=3)

    @patch("builtins.print")
    def test_start_epoch_verbose_1(self, mock_print: MagicMock) -> None:
        """Test start_epoch with verbose=1."""
        progress = Progress(verbose=1, total_epochs=5)
        progress.start_epoch(2)

        mock_print.assert_called_once_with("Epoch 3/5")

    @patch("builtins.print")
    def test_start_epoch_verbose_0(self, mock_print: MagicMock) -> None:
        """Test start_epoch with verbose=0 (no output)."""
        progress = Progress(verbose=0)
        progress.start_epoch(1)

        mock_print.assert_not_called()

    @patch("builtins.print")
    @patch("time.time")
    def test_end_epoch_verbose_2_with_val_loss(
        self, mock_time: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test end_epoch with verbose=2 and validation loss."""
        mock_time.side_effect = [0, 10, 15]  # constructor, start_epoch, end_epoch

        progress = Progress(verbose=2, total_epochs=3)
        progress.start_epoch(0)
        progress.end_epoch(0.4, 0.2)

        expected_call = "Epoch 1/3 - Train Loss: 0.4000, Val Loss: 0.2000 (5.00s)"
        mock_print.assert_called_with(expected_call)

    @patch("builtins.print")
    @patch("time.time")
    def test_end_epoch_verbose_2_without_val_loss(
        self, mock_time: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test end_epoch with verbose=2 and no validation loss."""
        mock_time.side_effect = [0, 10, 12]  # constructor, start_epoch, end_epoch

        progress = Progress(verbose=2, total_epochs=2)
        progress.start_epoch(1)
        progress.end_epoch(0.3)

        expected_call = "Epoch 2/2 - Train Loss: 0.3000 (2.00s)"
        mock_print.assert_called_with(expected_call)

    @patch("builtins.print")
    def test_end_epoch_verbose_0(self, mock_print: MagicMock) -> None:
        """Test end_epoch with verbose=0 (no output)."""
        progress = Progress(verbose=0)
        progress.end_epoch(0.5)
        mock_print.assert_not_called()

    @patch("builtins.print")
    @patch("time.time")
    def test_end_training_verbose_2(
        self, mock_time: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test end_training with verbose=2."""
        mock_time.side_effect = [0, 25]  # constructor, end_training

        progress = Progress(verbose=2)
        progress.end_training()

        mock_print.assert_called_with("Training completed in 25.00s")

    @patch("builtins.print")
    def test_end_training_verbose_0(self, mock_print: MagicMock) -> None:
        """Test end_training with verbose=0 (no output)."""
        progress = Progress(verbose=0)
        progress.end_training()
        mock_print.assert_not_called()

    @patch("torch_batteries.utils.progress.progress.tqdm")
    def test_start_phase_with_progress_bar(self, mock_tqdm: MagicMock) -> None:
        """Test start_phase creates progress bar when verbose=1."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = Progress(verbose=1)
        progress.start_phase(Phase.TRAIN, total_batches=10)

        mock_tqdm.assert_called_once_with(
            total=10,
            desc="Train",
            ncols=80,
            bar_format="{desc}: {n}/{total} {bar} {percentage:3.0f}%{postfix}",
            leave=True,
        )

    def test_start_phase_no_progress_bar_verbose_0(self) -> None:
        """Test start_phase doesn't create progress bar when verbose=0."""
        with patch("torch_batteries.utils.progress.progress.tqdm") as mock_tqdm:
            progress = Progress(verbose=0)
            progress.start_phase(Phase.TRAIN, total_batches=10)
            mock_tqdm.assert_not_called()

    def test_start_phase_no_progress_bar_zero_batches(self) -> None:
        """Test start_phase doesn't create progress bar when total_batches=0."""
        with patch("torch_batteries.utils.progress.progress.tqdm") as mock_tqdm:
            progress = Progress(verbose=1)
            progress.start_phase(Phase.TRAIN, total_batches=0)
            mock_tqdm.assert_not_called()

    @patch("torch_batteries.utils.progress.progress.tqdm")
    def test_update_with_metrics(self, mock_tqdm: MagicMock) -> None:
        """Test update method with metrics and batch_size."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = Progress(verbose=1)
        progress.start_phase(Phase.TRAIN, total_batches=5)
        metrics: ProgressMetrics = {"loss": 0.5}

        progress.update(metrics, 32)

        mock_pbar.set_postfix_str.assert_called_with("loss: 0.5000")
        mock_pbar.update.assert_called_with(1)

    @patch("torch_batteries.utils.progress.progress.tqdm")
    def test_update_validation_phase(self, mock_tqdm: MagicMock) -> None:
        """Test update method shows val_loss for validation phase."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = Progress(verbose=1)
        progress.start_phase(Phase.VALIDATION, total_batches=3)
        metrics: ProgressMetrics = {"loss": 0.3}

        progress.update(metrics, 16)

        mock_pbar.set_postfix_str.assert_called_with("val_loss: 0.3000")

    @patch("torch_batteries.utils.progress.progress.tqdm")
    def test_update_without_metrics(self, mock_tqdm: MagicMock) -> None:
        """Test update method without metrics."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = Progress(verbose=1)
        progress.start_phase(Phase.TRAIN, total_batches=3)
        progress.update()

        mock_pbar.set_postfix_str.assert_not_called()
        mock_pbar.update.assert_called_with(1)

    @patch("torch_batteries.utils.progress.progress.tqdm")
    def test_update_accumulates_loss(self, mock_tqdm: MagicMock) -> None:
        """Test that update accumulates loss correctly."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = Progress(verbose=1)
        progress.start_phase(Phase.TRAIN, total_batches=2)

        # First update
        progress.update({"loss": 0.5}, 10)
        # Second update
        progress.update({"loss": 0.3}, 20)

        # Should show average loss: (0.5*10 + 0.3*20) / (10+20) = 11/30 ≈ 0.3667
        expected_avg = (0.5 * 10 + 0.3 * 20) / (10 + 20)
        mock_pbar.set_postfix_str.assert_called_with(f"loss: {expected_avg:.4f}")

    @patch("torch_batteries.utils.progress.progress.tqdm")
    def test_end_phase_returns_average_loss(self, mock_tqdm: MagicMock) -> None:
        """Test that end_phase returns correct average loss."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = Progress(verbose=1)
        progress.start_phase(Phase.TRAIN, total_batches=2)

        progress.update({"loss": 0.4}, 10)
        progress.update({"loss": 0.6}, 20)

        avg_loss = progress.end_phase()

        # Expected: (0.4*10 + 0.6*20) / (10+20) = 16/30 ≈ 0.5333
        expected_avg = (0.4 * 10 + 0.6 * 20) / 30
        assert abs(avg_loss - expected_avg) < 1e-6
        mock_pbar.close.assert_called_once()

    @patch("torch_batteries.utils.progress.progress.tqdm")
    def test_end_phase_no_samples_returns_zero(self, mock_tqdm: MagicMock) -> None:
        """Test that end_phase returns 0 when no samples processed."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = Progress(verbose=1)
        progress.start_phase(Phase.TRAIN, total_batches=1)
        avg_loss = progress.end_phase()

        assert avg_loss == 0.0
        mock_pbar.close.assert_called_once()

    def test_end_phase_no_progress_bar(self) -> None:
        """Test end_phase when no progress bar was created."""
        progress = Progress(verbose=0)
        progress.start_phase(Phase.TRAIN, total_batches=10)
        avg_loss = progress.end_phase()
        assert avg_loss == 0.0  # No error should occur
