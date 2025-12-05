"""Tests for torch_batteries.trainer module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries.events import Event, charge
from torch_batteries.trainer import Battery


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # type: ignore[no-any-return]

    @charge(Event.TRAIN_STEP)
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        return nn.functional.mse_loss(pred, y)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        return nn.functional.mse_loss(pred, y)

    @charge(Event.TEST_STEP)
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        return nn.functional.mse_loss(pred, y)

    @charge(Event.PREDICT_STEP)
    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x)  # type: ignore[no-any-return]


class TestBattery:
    """Test cases for Battery trainer class."""

    def create_simple_data_loader(
        self, batch_size: int = 4, num_samples: int = 16, input_size: int = 10
    ) -> DataLoader:
        """Create a simple data loader for testing."""
        x = torch.randn(num_samples, input_size)
        y = torch.randn(num_samples, 1)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size)

    def test_init_with_defaults(self) -> None:
        """Test Battery initialization with default values."""
        model = SimpleModel()
        battery = Battery(model)

        assert battery.model is model
        assert isinstance(battery.device, torch.device)
        assert battery.optimizer is None

    def test_init_with_custom_device(self) -> None:
        """Test Battery initialization with custom device."""
        model = SimpleModel()
        battery = Battery(model, device="cpu")

        assert battery.device.type == "cpu"

    def test_init_with_optimizer(self) -> None:
        """Test Battery initialization with optimizer."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer)

        assert battery.optimizer is optimizer

    def test_optimizer_property_setter(self) -> None:
        """Test optimizer property setter."""
        model = SimpleModel()
        battery = Battery(model)
        optimizer = optim.Adam(model.parameters())

        battery.optimizer = optimizer
        assert battery.optimizer is optimizer

    @patch("torch_batteries.trainer.core.get_device")
    def test_auto_device_detection(self, mock_get_device: MagicMock) -> None:
        """Test auto device detection."""
        mock_device = torch.device("cpu")
        mock_get_device.return_value = mock_device

        model = SimpleModel()
        battery = Battery(model, device="auto")

        mock_get_device.assert_called_once_with("auto")
        assert battery.device == mock_device

    def test_model_moved_to_device(self) -> None:
        """Test that model is moved to specified device."""
        model = SimpleModel()
        battery = Battery(model, device="cpu")

        # Check that model parameters are on correct device
        for param in battery.model.parameters():
            assert param.device.type == "cpu"

    def test_train_without_optimizer_raises_error(self) -> None:
        """Test that training without optimizer raises ValueError."""
        model = SimpleModel()
        battery = Battery(model)
        train_loader = self.create_simple_data_loader()

        with pytest.raises(ValueError, match="Optimizer is required for training"):
            battery.train(train_loader)

    def test_train_without_training_step_raises_error(self) -> None:
        """Test that training without training step handler raises ValueError."""

        class ModelWithoutTrainStep(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)  # Match input size

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)  # type: ignore[no-any-return]

        model = ModelWithoutTrainStep()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer)
        train_loader = self.create_simple_data_loader()

        with pytest.raises(
            ValueError, match=r"No method decorated with @charge\(Event.TRAIN_STEP\)"
        ):
            battery.train(train_loader)

    @patch("torch_batteries.utils.progress.ProgressTracker")
    @patch("torch_batteries.utils.progress.EpochProgressTracker")
    def test_train_basic_functionality(
        self, mock_epoch_tracker: MagicMock, mock_progress_tracker: MagicMock
    ) -> None:
        """Test basic train functionality."""
        # Setup mocks
        mock_progress = MagicMock()
        mock_progress_tracker.return_value = mock_progress
        mock_epoch_progress = MagicMock()
        mock_epoch_progress.close.return_value = 0.5
        mock_epoch_tracker.return_value = mock_epoch_progress

        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer)
        train_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        result = battery.train(train_loader, epochs=2, verbose=0)

        assert isinstance(result, dict)
        assert "train_loss" in result
        assert "val_loss" in result
        assert len(result["train_loss"]) == 2  # 2 epochs
        assert len(result["val_loss"]) == 0  # no validation loader

    @patch("torch_batteries.utils.progress.ProgressTracker")
    @patch("torch_batteries.utils.progress.EpochProgressTracker")
    def test_train_with_validation(
        self, mock_epoch_tracker: MagicMock, mock_progress_tracker: MagicMock
    ) -> None:
        """Test training with validation loader."""
        # Setup mocks
        mock_progress = MagicMock()
        mock_progress_tracker.return_value = mock_progress
        mock_epoch_progress = MagicMock()
        mock_epoch_progress.close.return_value = 0.3
        mock_epoch_tracker.return_value = mock_epoch_progress

        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        battery = Battery(model, optimizer=optimizer)

        train_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)
        val_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        result = battery.train(train_loader, val_loader, epochs=1, verbose=0)

        assert len(result["train_loss"]) == 1
        assert len(result["val_loss"]) == 1

    def test_validate_epoch_without_handler_raises_error(self) -> None:
        """Test validation without validation step handler raises error."""

        class ModelWithoutValidationStep(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)  # Match input size\n

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)  # type: ignore[no-any-return]

            @charge(Event.TRAIN_STEP)
            def training_step(
                self, batch: tuple[torch.Tensor, torch.Tensor]
            ) -> torch.Tensor:
                x, y = batch
                pred = self(x)
                return nn.functional.mse_loss(pred, y)

        model = ModelWithoutValidationStep()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer)

        train_loader = self.create_simple_data_loader()
        val_loader = self.create_simple_data_loader()

        with pytest.raises(
            ValueError,
            match=r"No method decorated with @charge\(Event.VALIDATION_STEP\)",
        ):
            battery.train(train_loader, val_loader, epochs=1)

    def test_test_without_handler_raises_error(self) -> None:
        """Test testing without test step handler raises ValueError."""

        class ModelWithoutTestStep(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = ModelWithoutTestStep()
        battery = Battery(model)
        test_loader = self.create_simple_data_loader()

        with pytest.raises(
            ValueError, match=r"No method decorated with @charge\(Event.TEST_STEP\)"
        ):
            battery.test(test_loader)

    @patch("torch_batteries.utils.progress.EpochProgressTracker")
    def test_test_basic_functionality(self, mock_epoch_tracker: MagicMock) -> None:
        """Test basic test functionality."""
        mock_epoch_progress = MagicMock()
        mock_epoch_progress.close.return_value = 0.25
        mock_epoch_tracker.return_value = mock_epoch_progress

        model = SimpleModel()
        battery = Battery(model)
        test_loader = self.create_simple_data_loader()

        result = battery.test(test_loader, verbose=0)

        assert isinstance(result, dict)
        assert "test_loss" in result
        # Don't assert exact value since it's computed, just assert it's a float
        assert isinstance(result["test_loss"], float)

    def test_predict_without_handler_raises_error(self) -> None:
        """Test prediction without predict step handler raises ValueError."""

        class ModelWithoutPredictStep(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = ModelWithoutPredictStep()
        battery = Battery(model)
        data_loader = self.create_simple_data_loader()

        with pytest.raises(
            ValueError, match=r"No method decorated with @charge\(Event.PREDICT_STEP\)"
        ):
            battery.predict(data_loader)

    @patch("torch_batteries.utils.progress.EpochProgressTracker")
    def test_predict_basic_functionality(self, mock_epoch_tracker: MagicMock) -> None:
        """Test basic predict functionality."""
        mock_epoch_progress = MagicMock()
        mock_epoch_progress.close.return_value = 0.0
        mock_epoch_tracker.return_value = mock_epoch_progress

        model = SimpleModel()
        battery = Battery(model)
        data_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        result = battery.predict(data_loader, verbose=0)

        assert isinstance(result, dict)
        assert "predictions" in result
        assert len(result["predictions"]) == 2  # 4 samples / 2 batch_size = 2 batches

    def test_train_sets_model_to_train_mode(self) -> None:
        """Test that training sets model to train mode."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer)

        # Set model to eval mode initially
        model.eval()
        assert not model.training

        train_loader = self.create_simple_data_loader(batch_size=2, num_samples=2)

        with (
            patch("torch_batteries.utils.progress.ProgressTracker"),
            patch("torch_batteries.utils.progress.EpochProgressTracker") as mock_epoch,
        ):
            mock_epoch.return_value.close.return_value = 0.1
            battery.train(train_loader, epochs=1, verbose=0)

        assert model.training

    def test_test_sets_model_to_eval_mode(self) -> None:
        """Test that testing sets model to eval mode."""
        model = SimpleModel()
        battery = Battery(model)

        # Set model to train mode initially
        model.train()
        assert model.training

        test_loader = self.create_simple_data_loader(batch_size=2, num_samples=2)

        with patch("torch_batteries.utils.progress.EpochProgressTracker") as mock_epoch:
            mock_epoch.return_value.close.return_value = 0.1
            battery.test(test_loader, verbose=0)

        assert not model.training

    def test_predict_sets_model_to_eval_mode(self) -> None:
        """Test that prediction sets model to eval mode."""
        model = SimpleModel()
        battery = Battery(model)

        # Set model to train mode initially
        model.train()
        assert model.training

        data_loader = self.create_simple_data_loader(batch_size=2, num_samples=2)

        with patch("torch_batteries.utils.progress.EpochProgressTracker") as mock_epoch:
            mock_epoch.return_value.close.return_value = 0.0
            battery.predict(data_loader, verbose=0)

        assert not model.training

    def test_integration_train_and_test(self) -> None:
        """Test integration between training and testing."""
        model = SimpleModel(input_size=5, output_size=1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        battery = Battery(model, optimizer=optimizer)

        # Create simple data
        train_loader = self.create_simple_data_loader(
            batch_size=2, num_samples=4, input_size=5
        )
        test_loader = self.create_simple_data_loader(
            batch_size=2, num_samples=4, input_size=5
        )

        # Train for 1 epoch
        train_result = battery.train(train_loader, epochs=1, verbose=0)

        # Test the trained model
        test_result = battery.test(test_loader, verbose=0)

        assert isinstance(train_result["train_loss"][0], float)
        assert isinstance(test_result["test_loss"], float)
