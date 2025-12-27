"""Tests for torch_batteries.utils.metrics module."""

# mypy: disable-error-code="arg-type"
# ruff: noqa: N812, TRY003, EM101, TC003

from collections.abc import Callable
from unittest.mock import patch

import torch
import torch.nn.functional as F

from torch_batteries.utils.metrics import calculate_metrics


class TestCalculateMetrics:
    """Test cases for calculate_metrics function."""

    def test_single_metric_tensor_return(self) -> None:
        """Test calculation with a single metric that returns a tensor."""

        def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.l1_loss(pred, target)

        metrics = {"mae": mae}
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[1.1], [2.2], [2.9]])

        results = calculate_metrics(metrics, pred, target)

        assert "mae" in results
        assert isinstance(results["mae"], float)
        assert abs(results["mae"] - 0.1333) < 0.01  # Approximate check

    def test_single_metric_float_return(self) -> None:
        """Test calculation with a single metric that returns a float."""

        def custom_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
            return float((pred - target).abs().mean())

        metrics = {"custom": custom_metric}
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])

        results = calculate_metrics(metrics, pred, target)

        assert "custom" in results
        assert isinstance(results["custom"], float)
        assert results["custom"] == 0.5

    def test_multiple_metrics(self) -> None:
        """Test calculation with multiple metrics."""

        def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.l1_loss(pred, target)

        def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(pred, target)

        def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(F.mse_loss(pred, target))

        metrics = {"mae": mae, "mse": mse, "rmse": rmse}
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[1.0], [2.0], [3.0]])

        results = calculate_metrics(metrics, pred, target)

        assert len(results) == 3
        assert "mae" in results
        assert "mse" in results
        assert "rmse" in results
        assert results["mae"] == 0.0
        assert results["mse"] == 0.0
        assert results["rmse"] == 0.0

    def test_empty_metrics_dict(self) -> None:
        """Test calculation with an empty metrics dictionary."""
        metrics: dict[
            str, Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]
        ] = {}
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])

        results = calculate_metrics(metrics, pred, target)

        assert results == {}

    def test_metric_with_exception(self) -> None:
        """Test that failing metrics are skipped and logged."""

        def working_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
            return 1.0

        def failing_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
            raise ValueError("Intentional error")

        metrics = {"working": working_metric, "failing": failing_metric}
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])

        with patch("torch_batteries.utils.metrics.logger.warning") as mock_warning:
            results = calculate_metrics(metrics, pred, target)

            assert "working" in results
            assert "failing" not in results
            assert results["working"] == 1.0
            mock_warning.assert_called_once()
            assert "failing" in mock_warning.call_args[0][1]

    def test_all_metrics_fail(self) -> None:
        """Test when all metrics fail."""

        def failing_metric_1(pred: torch.Tensor, target: torch.Tensor) -> float:
            raise RuntimeError("Error 1")

        def failing_metric_2(pred: torch.Tensor, target: torch.Tensor) -> float:
            raise ValueError("Error 2")

        metrics = {"metric1": failing_metric_1, "metric2": failing_metric_2}
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])

        with patch("torch_batteries.utils.metrics.logger.warning") as mock_warning:
            results = calculate_metrics(metrics, pred, target)

            assert results == {}
            assert mock_warning.call_count == 2

    def test_tensor_scalar_conversion(self) -> None:
        """Test that tensor scalars are properly converted to float."""

        def tensor_scalar_metric(
            pred: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            return torch.tensor(42.5)

        metrics = {"tensor_scalar": tensor_scalar_metric}
        pred = torch.tensor([1.0])
        target = torch.tensor([1.0])

        results = calculate_metrics(metrics, pred, target)

        assert "tensor_scalar" in results
        assert isinstance(results["tensor_scalar"], float)
        assert results["tensor_scalar"] == 42.5

    def test_different_tensor_shapes(self) -> None:
        """Test metrics with different tensor shapes."""

        def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.l1_loss(pred, target)

        metrics = {"mae": mae}

        # Test with 1D tensors
        pred_1d = torch.tensor([1.0, 2.0, 3.0])
        target_1d = torch.tensor([1.5, 2.5, 3.5])
        results_1d = calculate_metrics(metrics, pred_1d, target_1d)
        assert "mae" in results_1d
        assert results_1d["mae"] == 0.5

        # Test with 2D tensors
        pred_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_2d = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        results_2d = calculate_metrics(metrics, pred_2d, target_2d)
        assert "mae" in results_2d
        assert results_2d["mae"] == 0.5

        # Test with 3D tensors
        pred_3d = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        target_3d = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        results_3d = calculate_metrics(metrics, pred_3d, target_3d)
        assert "mae" in results_3d
        assert results_3d["mae"] == 0.0

    def test_integer_return_conversion(self) -> None:
        """Test that integer returns are converted to float."""

        def int_metric(pred: torch.Tensor, target: torch.Tensor) -> int:
            return 10

        metrics = {"int_metric": int_metric}
        pred = torch.tensor([1.0])
        target = torch.tensor([1.0])

        results = calculate_metrics(metrics, pred, target)

        assert "int_metric" in results
        assert isinstance(results["int_metric"], float)
        assert results["int_metric"] == 10.0

    def test_metric_with_reduction_none(self) -> None:
        """Test metric that returns multiple values (should fail and be skipped)."""

        def unreduced_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # Returns a tensor with multiple values - .item() will fail
            return F.l1_loss(pred, target, reduction="none")

        metrics = {"unreduced": unreduced_metric}
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])

        with patch("torch_batteries.utils.metrics.logger.warning") as mock_warning:
            results = calculate_metrics(metrics, pred, target)

            # Should be skipped due to .item() error on multi-element tensor
            assert "unreduced" not in results
            mock_warning.assert_called_once()

    def test_accuracy_metric_example(self) -> None:
        """Test a classification accuracy metric."""

        def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
            pred_labels = pred.argmax(dim=1)
            correct = (pred_labels == target).sum().item()
            total = target.size(0)
            return correct / total if total > 0 else 0.0

        metrics = {"accuracy": accuracy}
        pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        target = torch.tensor([1, 0, 1, 0])

        results = calculate_metrics(metrics, pred, target)

        assert "accuracy" in results
        assert results["accuracy"] == 1.0  # All predictions correct

    def test_zero_division_handling(self) -> None:
        """Test metric that might cause division by zero is handled."""

        def potentially_problematic_metric(
            pred: torch.Tensor, target: torch.Tensor
        ) -> float:
            # This could cause issues with zero tensors
            denominator = target.sum()
            if denominator == 0:
                raise ValueError("Cannot divide by zero")
            return (pred.sum() / denominator).item()

        metrics = {"ratio": potentially_problematic_metric}
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0, 0.0, 0.0])

        with patch("torch_batteries.utils.metrics.logger.warning") as mock_warning:
            results = calculate_metrics(metrics, pred, target)

            assert "ratio" not in results
            mock_warning.assert_called_once()

    def test_large_batch_metrics(self) -> None:
        """Test metrics with large batch sizes."""

        def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(pred, target)

        metrics = {"mse": mse}
        batch_size = 1000
        pred = torch.randn(batch_size, 10)
        target = pred + 0.1  # Small difference

        results = calculate_metrics(metrics, pred, target)

        assert "mse" in results
        assert isinstance(results["mse"], float)
        assert results["mse"] > 0  # Should have some error
