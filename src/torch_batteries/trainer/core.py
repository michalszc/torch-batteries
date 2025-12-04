"""Battery trainer class for torch-batteries."""

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_batteries.events import Event, EventHandler
from torch_batteries.trainer.types import PredictResult, TestResult, TrainResult
from torch_batteries.utils.batch import get_batch_size
from torch_batteries.utils.logging import get_logger

logger = get_logger("trainer")


class Battery:
    """
    A flexible trainer class that uses decorated methods to define training behavior.

    The Battery class discovers methods decorated with @charge(Event.*) to automatically
    configure training, validation, testing, and prediction workflows.

    Args:
        model: PyTorch model (nn.Module)
        device: PyTorch device (cpu, cuda, etc.)
        optimizer: Optional optimizer for training

    Example:
        ```python
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

            @charge(Event.TRAIN_STEP)
            def training_step(self, batch):
                x, y = batch
                pred = self(x)
                loss = F.mse_loss(pred, y)
                return loss

            @charge(Event.VALIDATION_STEP)
            def validation_step(self, batch):
                x, y = batch
                pred = self(x)
                loss = F.mse_loss(pred, y)
                return loss

        battery = Battery(model, device='cuda', optimizer=optimizer)
        battery.train(train_loader, val_loader, epochs=10)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self.model = model.to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.optimizer = optimizer

        self.event_handler = EventHandler(self.model)

    def _move_batch_to_device(self, batch: Any) -> Any:
        """Move batch to device handling different data types."""
        if isinstance(batch, (list, tuple)):
            return [
                x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
            ]
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 1,
    ) -> TrainResult:
        """
        Train the model for the specified number of epochs.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs

        Returns:
            TrainResult containing training and validation metrics

        Raises:
            ValueError: If no training step handler is found
        """
        if not self.event_handler.has_handler(Event.TRAIN_STEP):
            msg = (
                "No method decorated with @charge(Event.TRAIN_STEP) found. "
                "Please add a training step method to your model."
            )
            raise ValueError(msg)

        if self.optimizer is None:
            msg = "Optimizer is required for training."
            raise ValueError(msg)

        metrics: TrainResult = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(epochs):
            # Training epoch
            train_loss = self._train_epoch(train_loader)
            metrics["train_loss"].append(train_loss)

            # Validation epoch
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                metrics["val_loss"].append(val_loss)

                logger.info(
                    "Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f",
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss,
                )
            else:
                logger.info(
                    "Epoch %d/%d - Train Loss: %.4f",
                    epoch + 1,
                    epochs,
                    train_loss,
                )

        return metrics

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_data in dataloader:
            batch = self._move_batch_to_device(batch_data)

            # Optimizer is guaranteed to be non-None by train() method
            self.optimizer.zero_grad()  # type: ignore[union-attr]

            # Forward pass
            loss = self.event_handler.call(Event.TRAIN_STEP, batch)
            assert loss is not None, "Training step must return a loss value."

            loss.backward()
            self.optimizer.step()  # type: ignore[union-attr]

            num_samples = get_batch_size(batch)

            total_loss += loss.item() * num_samples
            total_samples += num_samples

        return total_loss / total_samples

    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """Run a single validation epoch."""
        if not self.event_handler.has_handler(Event.VALIDATION_STEP):
            msg = (
                "No method decorated with @charge(Event.VALIDATION_STEP) found. "
                "Please add a validation step method to your model."
            )
            raise ValueError(msg)

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_data in dataloader:
                batch = self._move_batch_to_device(batch_data)

                loss = self.event_handler.call(Event.VALIDATION_STEP, batch)
                assert loss is not None, "Validation step must return a loss value."

                num_samples = get_batch_size(batch)
                total_loss += loss.item() * num_samples
                total_samples += num_samples

        return total_loss / total_samples

    def test(self, test_loader: DataLoader) -> TestResult:
        """
        Test the model on the provided data loader.

        Args:
            test_loader: Test data loader

        Returns:
            TestResult containing test loss

        Raises:
            ValueError: If no test step handler is found
        """
        if not self.event_handler.has_handler(Event.TEST_STEP):
            msg = (
                "No method decorated with @charge(Event.TEST_STEP) found. "
                "Please add a test step method to your model."
            )
            raise ValueError(msg)

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_data in test_loader:
                batch = self._move_batch_to_device(batch_data)

                loss = self.event_handler.call(Event.TEST_STEP, batch)
                assert loss is not None, "Test step must return a loss value."

                num_samples = get_batch_size(batch)
                total_loss += loss.item() * num_samples
                total_samples += num_samples

        test_loss = total_loss / total_samples
        return {"test_loss": [test_loss]}

    def predict(self, data_loader: DataLoader) -> PredictResult:
        """
        Generate predictions using the model.

        Args:
            data_loader: Data loader for prediction

        Returns:
            PredictResult containing predictions

        Raises:
            ValueError: If no predict step handler is found
        """
        if not self.event_handler.has_handler(Event.PREDICT_STEP):
            msg = (
                "No method decorated with @charge(Event.PREDICT_STEP) found. "
                "Please add a predict step method to your model."
            )
            raise ValueError(msg)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_data in data_loader:
                batch = self._move_batch_to_device(batch_data)

                prediction = self.event_handler.call(Event.PREDICT_STEP, batch)
                if prediction is not None:
                    predictions.append(prediction)

        return {"predictions": predictions}
