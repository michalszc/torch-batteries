"""Tests for torch_batteries.utils.progress module."""

from unittest.mock import MagicMock, patch

from torch_batteries.utils.progress import (
    EpochProgressTracker,
    Phase,
    ProgressMetrics,
    ProgressTracker,
)


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


class TestProgressTracker:
    """Test cases for ProgressTracker class."""

    def test_init_default_values(self) -> None:
        """Test ProgressTracker initialization with default values."""
        tracker = ProgressTracker()
        assert tracker.verbose == 1
        assert tracker.total_epochs == 1
        assert tracker.current_epoch == 0
        assert tracker.train_loss == 0.0
        assert tracker.val_loss is None

    def test_init_custom_values(self) -> None:
        """Test ProgressTracker initialization with custom values."""
        tracker = ProgressTracker(verbose=2, total_epochs=10)
        assert tracker.verbose == 2
        assert tracker.total_epochs == 10

    def test_property_setters(self) -> None:
        """Test property setters work correctly."""
        tracker = ProgressTracker()

        tracker.train_loss = 0.5
        assert tracker.train_loss == 0.5

        tracker.val_loss = 0.3
        assert tracker.val_loss == 0.3

        tracker.val_loss = None
        assert tracker.val_loss is None

    @patch("builtins.print")
    def test_start_epoch_verbose_1(self, mock_print: MagicMock) -> None:
        """Test start_epoch with verbose=1."""
        tracker = ProgressTracker(verbose=1, total_epochs=5)
        tracker.start_epoch(2)

        assert tracker.current_epoch == 2
        mock_print.assert_called_once_with("Epoch 3/5")

    @patch("builtins.print")
    def test_start_epoch_verbose_0(self, mock_print: MagicMock) -> None:
        """Test start_epoch with verbose=0 (no output)."""
        tracker = ProgressTracker(verbose=0)
        tracker.start_epoch(1)

        assert tracker.current_epoch == 1
        mock_print.assert_not_called()

    @patch("builtins.print")
    @patch("time.time")
    def test_end_epoch_verbose_2_with_val_loss(
        self, mock_time: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test end_epoch with verbose=2 and validation loss."""
        # constructor (2 calls), epoch_start, end_epoch
        mock_time.side_effect = [0, 5, 10, 15]

        tracker = ProgressTracker(verbose=2, total_epochs=3)
        tracker.start_epoch(0)
        tracker.train_loss = 0.4
        tracker.val_loss = 0.2

        tracker.end_epoch()

        expected_call = "Epoch 1/3 - Train Loss: 0.4000, Val Loss: 0.2000 (5.00s)"
        mock_print.assert_called_with(expected_call)

    @patch("builtins.print")
    @patch("time.time")
    def test_end_epoch_verbose_2_without_val_loss(
        self, mock_time: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test end_epoch with verbose=2 and no validation loss."""
        # constructor (2 calls), epoch_start, end_epoch
        mock_time.side_effect = [0, 5, 10, 12]

        tracker = ProgressTracker(verbose=2, total_epochs=2)
        tracker.start_epoch(1)
        tracker.train_loss = 0.3

        tracker.end_epoch()

        expected_call = "Epoch 2/2 - Train Loss: 0.3000 (2.00s)"
        mock_print.assert_called_with(expected_call)

    @patch("builtins.print")
    def test_end_epoch_verbose_0(self, mock_print: MagicMock) -> None:
        """Test end_epoch with verbose=0 (no output)."""
        tracker = ProgressTracker(verbose=0)
        tracker.end_epoch()
        mock_print.assert_not_called()

    @patch("builtins.print")
    @patch("time.time")
    def test_end_phase_verbose_2(
        self, mock_time: MagicMock, mock_print: MagicMock
    ) -> None:
        """Test end_phase with verbose=2."""
        mock_time.side_effect = [0, 5, 25]  # constructor (2 calls), end_phase

        tracker = ProgressTracker(verbose=2)
        tracker.end_phase()

        mock_print.assert_called_with("Training completed in 25.00s")

    @patch("builtins.print")
    def test_end_phase_verbose_0(self, mock_print: MagicMock) -> None:
        """Test end_phase with verbose=0 (no output)."""
        tracker = ProgressTracker(verbose=0)
        tracker.end_phase()
        mock_print.assert_not_called()


class TestEpochProgressTracker:
    """Test cases for EpochProgressTracker class."""

    def test_init_default_values(self) -> None:
        """Test EpochProgressTracker initialization with default values."""
        tracker = EpochProgressTracker()
        # Can't access private attributes directly, but can test public behavior
        assert tracker is not None

    @patch("torch_batteries.utils.progress.epoch_progress_tracker.tqdm")
    def test_init_with_progress_bar(self, mock_tqdm: MagicMock) -> None:
        """Test initialization creates progress bar when verbose=1."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        EpochProgressTracker(verbose=1, phase=Phase.TRAIN, total_batches=10)

        mock_tqdm.assert_called_once_with(
            total=10,
            desc="Train",
            ncols=80,
            bar_format="{desc}: {n}/{total} {bar} {percentage:3.0f}%{postfix}",
            leave=True,
        )

    def test_init_no_progress_bar_verbose_0(self) -> None:
        """Test initialization doesn't create progress bar when verbose=0."""
        with patch(
            "torch_batteries.utils.progress.epoch_progress_tracker.tqdm"
        ) as mock_tqdm:
            EpochProgressTracker(verbose=0, total_batches=10)
            mock_tqdm.assert_not_called()

    def test_init_no_progress_bar_zero_batches(self) -> None:
        """Test initialization doesn't create progress bar when total_batches=0."""
        with patch(
            "torch_batteries.utils.progress.epoch_progress_tracker.tqdm"
        ) as mock_tqdm:
            EpochProgressTracker(verbose=1, total_batches=0)
            mock_tqdm.assert_not_called()

    @patch("torch_batteries.utils.progress.epoch_progress_tracker.tqdm")
    def test_update_with_metrics(self, mock_tqdm: MagicMock) -> None:
        """Test update method with metrics and batch_size."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tracker = EpochProgressTracker(verbose=1, phase=Phase.TRAIN, total_batches=5)
        metrics: ProgressMetrics = {"loss": 0.5}

        tracker.update(metrics, 32)

        mock_pbar.set_postfix_str.assert_called_with("loss: 0.5000")
        mock_pbar.update.assert_called_with(1)

    @patch("torch_batteries.utils.progress.epoch_progress_tracker.tqdm")
    def test_update_validation_phase(self, mock_tqdm: MagicMock) -> None:
        """Test update method shows val_loss for validation phase."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tracker = EpochProgressTracker(
            verbose=1, phase=Phase.VALIDATION, total_batches=3
        )
        metrics: ProgressMetrics = {"loss": 0.3}

        tracker.update(metrics, 16)

        mock_pbar.set_postfix_str.assert_called_with("val_loss: 0.3000")

    @patch("torch_batteries.utils.progress.epoch_progress_tracker.tqdm")
    def test_update_without_metrics(self, mock_tqdm: MagicMock) -> None:
        """Test update method without metrics."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tracker = EpochProgressTracker(verbose=1, total_batches=3)
        tracker.update()

        mock_pbar.set_postfix_str.assert_not_called()
        mock_pbar.update.assert_called_with(1)

    @patch("torch_batteries.utils.progress.epoch_progress_tracker.tqdm")
    def test_update_accumulates_loss(self, mock_tqdm: MagicMock) -> None:
        """Test that update accumulates loss correctly."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tracker = EpochProgressTracker(verbose=1, total_batches=2)

        # First update
        tracker.update({"loss": 0.5}, 10)
        # Second update
        tracker.update({"loss": 0.3}, 20)

        # Should show average loss: (0.5*10 + 0.3*20) / (10+20) = 11/30 ≈ 0.3667
        expected_avg = (0.5 * 10 + 0.3 * 20) / (10 + 20)
        mock_pbar.set_postfix_str.assert_called_with(f"loss: {expected_avg:.4f}")

    @patch("torch_batteries.utils.progress.epoch_progress_tracker.tqdm")
    def test_close_returns_average_loss(self, mock_tqdm: MagicMock) -> None:
        """Test that close returns correct average loss."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tracker = EpochProgressTracker(verbose=1, total_batches=2)

        tracker.update({"loss": 0.4}, 10)
        tracker.update({"loss": 0.6}, 20)

        avg_loss = tracker.close()

        # Expected: (0.4*10 + 0.6*20) / (10+20) = 16/30 ≈ 0.5333
        expected_avg = (0.4 * 10 + 0.6 * 20) / 30
        assert abs(avg_loss - expected_avg) < 1e-6
        mock_pbar.close.assert_called_once()

    @patch("torch_batteries.utils.progress.epoch_progress_tracker.tqdm")
    def test_close_no_samples_returns_zero(self, mock_tqdm: MagicMock) -> None:
        """Test that close returns 0 when no samples processed."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tracker = EpochProgressTracker(verbose=1, total_batches=1)
        avg_loss = tracker.close()

        assert avg_loss == 0.0
        mock_pbar.close.assert_called_once()

    def test_close_no_progress_bar(self) -> None:
        """Test close when no progress bar was created."""
        tracker = EpochProgressTracker(verbose=0)
        avg_loss = tracker.close()
        assert avg_loss == 0.0  # No error should occur
