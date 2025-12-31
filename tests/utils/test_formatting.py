"""Tests for torch_batteries.utils.formatting module."""

from torch_batteries.utils.formatting import format_metrics


class TestFormatMetrics:
    """Test cases for format_metrics function."""

    def test_single_metric(self) -> None:
        """Test formatting a single metric."""
        metrics = {"loss": 0.1234}
        result = format_metrics(metrics)
        assert result == "Loss: 0.1234"

    def test_multiple_metrics(self) -> None:
        """Test formatting multiple metrics."""
        metrics = {"loss": 0.1234, "mae": 0.5678}
        result = format_metrics(metrics)
        assert result == "Loss: 0.1234, Mae: 0.5678"

    def test_snake_case_conversion(self) -> None:
        """Test that snake_case names are converted to Title Case."""
        metrics = {"train_loss": 0.1234, "val_mae": 0.5678}
        result = format_metrics(metrics)
        assert result == "Train Loss: 0.1234, Val Mae: 0.5678"

    def test_with_prefix(self) -> None:
        """Test formatting with a prefix."""
        metrics = {"loss": 0.1234, "mae": 0.5678}
        result = format_metrics(metrics, prefix="Train ")
        assert result == "Train Loss: 0.1234, Train Mae: 0.5678"

    def test_empty_metrics(self) -> None:
        """Test formatting an empty metrics dictionary."""
        metrics: dict[str, float] = {}
        result = format_metrics(metrics)
        assert result == ""

    def test_decimal_precision(self) -> None:
        """Test that values are formatted to 4 decimal places."""
        metrics = {"loss": 0.123456789}
        result = format_metrics(metrics)
        assert result == "Loss: 0.1235"  # Rounded to 4 decimals

    def test_zero_value(self) -> None:
        """Test formatting zero values."""
        metrics = {"loss": 0.0}
        result = format_metrics(metrics)
        assert result == "Loss: 0.0000"

    def test_large_value(self) -> None:
        """Test formatting large values."""
        metrics = {"loss": 1234.5678}
        result = format_metrics(metrics)
        assert result == "Loss: 1234.5678"

    def test_negative_value(self) -> None:
        """Test formatting negative values."""
        metrics = {"loss": -0.1234}
        result = format_metrics(metrics)
        assert result == "Loss: -0.1234"

    def test_multiple_underscores(self) -> None:
        """Test names with multiple underscores."""
        metrics = {"train_val_test_loss": 0.1234}
        result = format_metrics(metrics)
        assert result == "Train Val Test Loss: 0.1234"

    def test_metrics_order_preserved(self) -> None:
        """Test that the order of metrics is preserved."""
        # Note: In Python 3.7+, dict order is guaranteed
        metrics = {"z_metric": 0.1, "a_metric": 0.2, "m_metric": 0.3}
        result = format_metrics(metrics)
        assert result == "Z Metric: 0.1000, A Metric: 0.2000, M Metric: 0.3000"

    def test_realistic_training_metrics(self) -> None:
        """Test formatting realistic training metrics."""
        metrics = {
            "loss": 0.4532,
            "mae": 0.2134,
            "rmse": 0.3456,
            "r2_score": 0.8765,
        }
        result = format_metrics(metrics, prefix="Train ")
        expected = (
            "Train Loss: 0.4532, "
            "Train Mae: 0.2134, "
            "Train Rmse: 0.3456, "
            "Train R2 Score: 0.8765"
        )
        assert result == expected

    def test_very_small_value(self) -> None:
        """Test formatting very small values."""
        metrics = {"loss": 0.0001234}
        result = format_metrics(metrics)
        assert result == "Loss: 0.0001"

    def test_scientific_notation_input(self) -> None:
        """Test that scientific notation values are formatted correctly."""
        metrics = {"loss": 1.234e-5}
        result = format_metrics(metrics)
        assert result == "Loss: 0.0000"  # Rounds to 4 decimals

    def test_mixed_case_names(self) -> None:
        """Test that mixed case names are handled correctly."""
        metrics = {"Loss": 0.1234, "MAE": 0.5678}
        result = format_metrics(metrics)
        # Title case applied after replacing underscores
        assert result == "Loss: 0.1234, Mae: 0.5678"

    def test_single_letter_metric(self) -> None:
        """Test formatting single letter metric names."""
        metrics = {"r": 0.1234}
        result = format_metrics(metrics)
        assert result == "R: 0.1234"

    def test_numeric_suffix(self) -> None:
        """Test metric names with numeric suffixes."""
        metrics = {"loss1": 0.1234, "loss2": 0.5678}
        result = format_metrics(metrics)
        assert result == "Loss1: 0.1234, Loss2: 0.5678"

    def test_empty_prefix(self) -> None:
        """Test that empty string prefix works same as no prefix."""
        metrics = {"loss": 0.1234}
        result_no_prefix = format_metrics(metrics)
        result_empty_prefix = format_metrics(metrics, prefix="")
        assert result_no_prefix == result_empty_prefix
