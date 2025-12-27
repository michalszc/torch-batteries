"""Tests for torch_batteries.utils.progress.BarProgress class."""

from unittest.mock import MagicMock, patch

from torch_batteries.utils.progress import BarProgress, Phase


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
        avg_metrics = progress.end_phase()

        mock_pbar.close.assert_called_once()
        assert isinstance(avg_metrics, dict)
        assert avg_metrics["loss"] == 0.4

    @patch("builtins.print")
    def test_end_epoch_no_output(self, mock_print: MagicMock) -> None:
        """Test end_epoch produces no output for verbose=1."""
        progress = BarProgress(total_epochs=1)
        progress.end_epoch()
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_end_training_no_output(self, mock_print: MagicMock) -> None:
        """Test end_training produces no output for verbose=1."""
        progress = BarProgress(total_epochs=1)
        progress.end_training()
        mock_print.assert_not_called()
