"""Battery trainer class for torch-batteries."""

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_batteries.events import Event, EventHandler
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

        # Create event handler to manage decorated methods
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
    ) -> dict[str, list[float]]:
        """
        Train the model for the specified number of epochs.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs

        Returns:
            Dictionary containing training and validation metrics

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

        metrics: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(epochs):
            # Training epoch
            self.event_handler.call(Event.TRAIN_EPOCH_START, epoch)
            train_loss = self._train_epoch(train_loader)
            self.event_handler.call(Event.TRAIN_EPOCH_END, epoch, train_loss)
            metrics["train_loss"].append(train_loss)

            # Validation epoch
            if val_loader:
                self.event_handler.call(Event.VALIDATION_EPOCH_START, epoch)
                val_loss = self._validate_epoch(val_loader)
                self.event_handler.call(Event.VALIDATION_EPOCH_END, epoch, val_loss)
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
        num_batches = 0

        for batch_data in dataloader:
            # Move batch to device
            batch = self._move_batch_to_device(batch_data)

            self.event_handler.call(Event.TRAIN_STEP_START, batch)
            # Forward pass
            loss = self.event_handler.call(Event.TRAIN_STEP, batch)

            if loss is not None:
                # Backward pass
                loss.backward()
                assert self.optimizer is not None  # Already checked in train method
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                num_batches += 1

            self.event_handler.call(Event.TRAIN_STEP_END, batch, loss)

        return total_loss / max(num_batches, 1)

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
        num_batches = 0

        with torch.no_grad():
            for batch_data in dataloader:
                # Move batch to device
                batch = self._move_batch_to_device(batch_data)

                self.event_handler.call(Event.VALIDATION_STEP_START, batch)

                loss = self.event_handler.call(Event.VALIDATION_STEP, batch)

                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1

                self.event_handler.call(Event.VALIDATION_STEP_END, batch, loss)

        return total_loss / max(num_batches, 1)

    def test(self, test_loader: DataLoader) -> dict[str, Any]:
        """
        Test the model on the provided data loader.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary containing test results

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
        results = []

        with torch.no_grad():
            for batch_data in test_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch_data)

                self.event_handler.call(Event.TEST_STEP_START, batch)

                result = self.event_handler.call(Event.TEST_STEP, batch)
                if result is not None:
                    results.append(result)

                self.event_handler.call(Event.TEST_STEP_END, batch, result)

        return {"results": results}

    def predict(self, data_loader: DataLoader) -> list[Any]:
        """
        Generate predictions using the model.

        Args:
            data_loader: Data loader for prediction

        Returns:
            List of predictions

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
                # Move batch to device
                batch = self._move_batch_to_device(batch_data)

                self.event_handler.call(Event.PREDICT_STEP_START, batch)

                prediction = self.event_handler.call(Event.PREDICT_STEP, batch)
                if prediction is not None:
                    predictions.append(prediction)

                self.event_handler.call(Event.PREDICT_STEP_END, batch, prediction)

        return predictions
