"""Tests for torch_batteries.utils.progress module."""

from unittest.mock import MagicMock, patch

import pytest

from torch_batteries.utils.progress import (
    BarProgress,
    Phase,
    ProgressFactory,
    ProgressMetrics,
    SilentProgress,
    SimpleProgress,
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


class TestProgressFactory:
    """Test cases for ProgressFactory class (main tests)."""

    def test_creates_silent_progress(self) -> None:
        """Test ProgressFactory creates SilentProgress for verbose=0."""
        progress = ProgressFactory.create(verbose=0, total_epochs=5)
        assert isinstance(progress, SilentProgress)

    def test_creates_progress_bar(self) -> None:
        """Test ProgressFactory creates BarProgress for verbose=1."""
        progress = ProgressFactory.create(verbose=1, total_epochs=5)
        assert isinstance(progress, BarProgress)

    def test_creates_simple_progress(self) -> None:
        """Test ProgressFactory creates SimpleProgress for verbose=2."""
        progress = ProgressFactory.create(verbose=2, total_epochs=5)
        assert isinstance(progress, SimpleProgress)

    def test_invalid_verbose_raises_error(self) -> None:
        """Test ProgressFactory raises error for invalid verbose level."""
        with pytest.raises(ValueError, match="Invalid verbose level"):
            ProgressFactory.create(verbose=3, total_epochs=1)

    def test_default_values(self) -> None:
        """Test ProgressFactory uses default values."""
        progress = ProgressFactory.create(verbose=1)  # Explicitly set default
        assert isinstance(progress, BarProgress)  # verbose=1 is default


class TestProgressFactoryCreate:
    """Test cases for ProgressFactory.create() static method."""

    def test_create_silent_progress(self) -> None:
        """Test factory creates SilentProgress for verbose=0."""
        progress = ProgressFactory.create(verbose=0, total_epochs=5)
        assert isinstance(progress, SilentProgress)

    def test_create_progress_bar(self) -> None:
        """Test factory creates BarProgress for verbose=1."""
        progress = ProgressFactory.create(verbose=1, total_epochs=5)
        assert isinstance(progress, BarProgress)

    def test_create_simple_progress(self) -> None:
        """Test factory creates SimpleProgress for verbose=2."""
        progress = ProgressFactory.create(verbose=2, total_epochs=5)
        assert isinstance(progress, SimpleProgress)

    def test_create_invalid_verbose(self) -> None:
        """Test factory raises error for invalid verbose level."""
        with pytest.raises(ValueError, match="Invalid verbose level"):
            ProgressFactory.create(verbose=3, total_epochs=1)


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
        progress.end_epoch(0.5, 0.3)
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_end_training_no_output(self, mock_print: MagicMock) -> None:
        """Test end_training produces no output."""
        progress = SilentProgress()
        progress.end_training()
        mock_print.assert_not_called()


class TestBarProgress:
    """Test cases for BarProgress class."""

    def test_init(self) -> None:
        """Test BarProgress initialization."""
        progress = BarProgress(total_epochs=5)
        assert progress is not None

    def test_start_epoch_stores_epoch_number(self) -> None:
        """Test start_epoch stores epoch number."""
        progress = BarProgress(total_epochs=5)
        progress.start_epoch(2)
        assert progress._current_epoch == 2  # noqa: SLF001

    @patch("torch_batteries.utils.progress.progress_bar.tqdm")
    def test_start_phase_creates_progress_bar(self, mock_tqdm: MagicMock) -> None:
        """Test start_phase creates progress bar."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = BarProgress(total_epochs=1)
        progress.start_epoch(0)
        progress.start_phase(Phase.TRAIN, total_batches=10)

        mock_tqdm.assert_called_once_with(
            total=10,
            desc="Epoch 1/1 [Train]",
            leave=True,
        )

    @patch("torch_batteries.utils.progress.progress_bar.tqdm")
    def test_start_phase_no_bar_zero_batches(self, mock_tqdm: MagicMock) -> None:
        """Test start_phase doesn't create bar for zero batches."""
        progress = BarProgress(total_epochs=1)
        progress.start_phase(Phase.TRAIN, total_batches=0)
        mock_tqdm.assert_not_called()

    @patch("torch_batteries.utils.progress.progress_bar.tqdm")
    def test_update_with_metrics(self, mock_tqdm: MagicMock) -> None:
        """Test update method with metrics."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = BarProgress(total_epochs=1)
        progress.start_phase(Phase.TRAIN, total_batches=5)
        progress.update({"loss": 0.5}, 32)

        mock_pbar.set_postfix_str.assert_called_with("Loss=0.5000")
        mock_pbar.update.assert_called_with(1)

    @patch("torch_batteries.utils.progress.progress_bar.tqdm")
    def test_update_validation_shows_loss(self, mock_tqdm: MagicMock) -> None:
        """Test update shows Loss for validation phase."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = BarProgress(total_epochs=1)
        progress.start_phase(Phase.VALIDATION, total_batches=3)
        progress.update({"loss": 0.3}, 16)

        mock_pbar.set_postfix_str.assert_called_with("Loss=0.3000")

    @patch("torch_batteries.utils.progress.progress_bar.tqdm")
    def test_end_phase_closes_bar(self, mock_tqdm: MagicMock) -> None:
        """Test end_phase closes progress bar."""
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        progress = BarProgress(total_epochs=1)
        progress.start_phase(Phase.TRAIN, total_batches=2)
        progress.update({"loss": 0.4}, 10)
        avg_loss = progress.end_phase()

        mock_pbar.close.assert_called_once()
        assert avg_loss == 0.4

    @patch("builtins.print")
    def test_end_epoch_no_output(self, mock_print: MagicMock) -> None:
        """Test end_epoch produces no output for verbose=1."""
        progress = BarProgress(total_epochs=1)
        progress.end_epoch(0.5, 0.3)
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_end_training_no_output(self, mock_print: MagicMock) -> None:
        """Test end_training produces no output for verbose=1."""
        progress = BarProgress(total_epochs=1)
        progress.end_training()
        mock_print.assert_not_called()


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
        progress.end_epoch(0.4, 0.2)

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
        progress.end_epoch(0.3)

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
