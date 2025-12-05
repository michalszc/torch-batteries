"""Tests for torch_batteries.trainer.types module."""

import torch

from torch_batteries.trainer.types import PredictResult, TestResult, TrainResult


class TestTrainerTypes:
    """Test cases for trainer type definitions."""

    def test_train_result_structure(self) -> None:
        """Test TrainResult type structure."""
        train_result: TrainResult = {
            "train_loss": [0.5, 0.3, 0.2],
            "val_loss": [0.4, 0.25, 0.15],
        }

        assert "train_loss" in train_result
        assert "val_loss" in train_result
        assert isinstance(train_result["train_loss"], list)
        assert isinstance(train_result["val_loss"], list)

    def test_train_result_empty_val_loss(self) -> None:
        """Test TrainResult with empty validation loss."""
        train_result: TrainResult = {
            "train_loss": [0.5, 0.3],
            "val_loss": [],
        }

        assert len(train_result["val_loss"]) == 0
        assert len(train_result["train_loss"]) == 2

    def test_test_result_structure(self) -> None:
        """Test TestResult type structure."""
        test_result: TestResult = {
            "test_loss": 0.25,
        }

        assert "test_loss" in test_result
        assert isinstance(test_result["test_loss"], float)

    def test_predict_result_structure(self) -> None:
        """Test PredictResult type structure."""
        predictions = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
        ]

        predict_result: PredictResult = {
            "predictions": predictions,
        }

        assert "predictions" in predict_result
        assert isinstance(predict_result["predictions"], list)
        assert len(predict_result["predictions"]) == 2

    def test_predict_result_empty_predictions(self) -> None:
        """Test PredictResult with empty predictions."""
        predict_result: PredictResult = {
            "predictions": [],
        }

        assert len(predict_result["predictions"]) == 0

    def test_predict_result_with_various_types(self) -> None:
        """Test PredictResult with various prediction types."""
        predictions = [
            torch.tensor([1.0]),
            torch.tensor([[2.0, 3.0]]),
            torch.tensor([[[4.0, 5.0, 6.0]]]),
        ]

        predict_result: PredictResult = {
            "predictions": predictions,
        }

        assert len(predict_result["predictions"]) == 3
        assert all(isinstance(p, torch.Tensor) for p in predict_result["predictions"])

    def test_train_result_type_compatibility(self) -> None:
        """Test that TrainResult is compatible with expected usage patterns."""

        def process_train_result(result: TrainResult) -> float:
            """Example function that processes train results."""
            return sum(result["train_loss"]) / len(result["train_loss"])

        train_result: TrainResult = {
            "train_loss": [0.8, 0.6, 0.4],
            "val_loss": [0.7, 0.5, 0.3],
        }

        avg_loss = process_train_result(train_result)
        assert abs(avg_loss - 0.6) < 1e-6

    def test_test_result_type_compatibility(self) -> None:
        """Test that TestResult is compatible with expected usage patterns."""

        def process_test_result(result: TestResult) -> str:
            """Example function that processes test results."""
            return f"Test loss: {result['test_loss']:.4f}"

        test_result: TestResult = {
            "test_loss": 0.1234,
        }

        message = process_test_result(test_result)
        assert message == "Test loss: 0.1234"

    def test_predict_result_type_compatibility(self) -> None:
        """Test that PredictResult is compatible with expected usage patterns."""

        def process_predictions(result: PredictResult) -> int:
            """Example function that processes predictions."""
            return len(result["predictions"])

        predictions = [torch.randn(5, 1) for _ in range(3)]
        predict_result: PredictResult = {
            "predictions": predictions,
        }

        count = process_predictions(predict_result)
        assert count == 3

    def test_result_types_with_additional_fields(self) -> None:
        """Test that result types can be extended with additional fields."""
        # These should work even though we add extra fields
        extended_train_result = {
            "train_loss": [0.5, 0.3],
            "val_loss": [0.4, 0.2],
            "epoch_times": [1.2, 1.1],  # Additional field
        }

        extended_test_result = {
            "test_loss": 0.15,
            "accuracy": 0.95,  # Additional field
        }

        extended_predict_result = {
            "predictions": [torch.tensor([1.0])],
            "confidences": [0.98],  # Additional field
        }

        # Type annotations should still work
        # (TypedDict with total=False allows extra keys)
        train_result: TrainResult = extended_train_result  # type: ignore[assignment]
        test_result: TestResult = extended_test_result  # type: ignore[assignment]
        predict_result: PredictResult = extended_predict_result  # type: ignore[assignment]

        assert "train_loss" in train_result
        assert "test_loss" in test_result
        assert "predictions" in predict_result
