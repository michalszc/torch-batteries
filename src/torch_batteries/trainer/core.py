"""Battery trainer class for torch-batteries."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_batteries.events import Event, EventHandler
from torch_batteries.trainer.types import PredictResult, TestResult, TrainResult
from torch_batteries.utils.batch import get_batch_size
from torch_batteries.utils.device import get_device, move_to_device
from torch_batteries.utils.logging import get_logger
from torch_batteries.utils.progress import Phase, Progress, ProgressFactory

logger = get_logger("trainer")


class Battery:
    """
    A flexible trainer class that uses decorated methods to define training behavior.

    The Battery class discovers methods decorated with @charge(Event.*) to automatically
    configure training, validation, testing, and prediction workflows.

    Args:
        model: PyTorch model (nn.Module)
        device: PyTorch device (cpu, cuda, etc.). If 'auto', detects available device.
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

        battery = Battery(model, optimizer=optimizer)  # Auto-detects device
        battery.train(train_loader, val_loader, epochs=10)
        ```
    """

    __slots__ = ("_device", "_event_handler", "_model", "_optimizer")

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "auto",
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self._device = get_device(device)
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._event_handler = EventHandler(self._model)

    @property
    def model(self) -> nn.Module:
        """Get the model."""
        return self._model

    @property
    def device(self) -> torch.device:
        """Get the device."""
        return self._device

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        """Get the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: torch.optim.Optimizer | None) -> None:
        """Set the optimizer."""
        self._optimizer = value

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 1,
        verbose: int = 1,
    ) -> TrainResult:
        """
        Train the model for the specified number of epochs.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            verbose: Verbosity level (0=silent, 1=progress bars, 2=epoch logs)

        Returns:
            TrainResult containing training and validation metrics

        Raises:
            ValueError: If no training step handler is found
        """
        if not self._event_handler.has_handler(Event.TRAIN_STEP):
            msg = (
                "No method decorated with @charge(Event.TRAIN_STEP) found. "
                "Please add a training step method to your model."
            )
            raise ValueError(msg)

        if self._optimizer is None:
            msg = "Optimizer is required for training."
            raise ValueError(msg)

        metrics: TrainResult = {
            "train_loss": [],
            "val_loss": [],
        }

        progress = ProgressFactory.create(verbose=verbose, total_epochs=epochs)

        for epoch in range(epochs):
            progress.start_epoch(epoch)

            train_loss = self._train_epoch(train_loader, progress)
            metrics["train_loss"].append(train_loss)

            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader, progress)
                metrics["val_loss"].append(val_loss)

            progress.end_epoch()

        progress.end_training()
        return metrics

    def _train_epoch(self, dataloader: DataLoader, progress: Progress) -> float:
        """Run a single training epoch.

        Args:
            dataloader: Training data loader
            progress: Progress tracker instance

        Returns:
            Average training loss for the epoch
        """
        self._model.train()

        progress.start_phase(Phase.TRAIN, total_batches=len(dataloader))

        for batch_data in dataloader:
            batch = move_to_device(batch_data, self._device)

            # Optimizer is guaranteed to be non-None by train() method
            self._optimizer.zero_grad()  # type: ignore[union-attr]

            loss = self._event_handler.call(Event.TRAIN_STEP, batch)
            assert loss is not None, "Training step must return a loss value."

            loss.backward()
            self._optimizer.step()  # type: ignore[union-attr]

            num_samples = get_batch_size(batch)
            progress.update({"loss": loss.item()}, num_samples)

        return progress.end_phase()

    def _validate_epoch(self, dataloader: DataLoader, progress: Progress) -> float:
        """Run a single validation epoch.

        Args:
            dataloader: Validation data loader
            progress: Progress tracker instance

        Returns:
            Average validation loss for the epoch
        """
        if not self._event_handler.has_handler(Event.VALIDATION_STEP):
            msg = (
                "No method decorated with @charge(Event.VALIDATION_STEP) found. "
                "Please add a validation step method to your model."
            )
            raise ValueError(msg)

        self._model.eval()

        progress.start_phase(Phase.VALIDATION, total_batches=len(dataloader))

        with torch.no_grad():
            for batch_data in dataloader:
                batch = move_to_device(batch_data, self._device)

                loss = self._event_handler.call(Event.VALIDATION_STEP, batch)
                assert loss is not None, "Validation step must return a loss value."

                num_samples = get_batch_size(batch)

                progress.update({"loss": loss.item()}, num_samples)

        return progress.end_phase()

    def test(self, test_loader: DataLoader, verbose: int = 1) -> TestResult:
        """
        Test the model on the provided data loader.

        Args:
            test_loader: Test data loader
            verbose: Verbosity level (0=silent, 1=progress bar, 2=simple log)

        Returns:
            TestResult containing test loss

        Raises:
            ValueError: If no test step handler is found
        """
        if not self._event_handler.has_handler(Event.TEST_STEP):
            msg = (
                "No method decorated with @charge(Event.TEST_STEP) found. "
                "Please add a test step method to your model."
            )
            raise ValueError(msg)

        self._model.eval()

        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(0)
        progress.start_phase(Phase.TEST, total_batches=len(test_loader))

        with torch.no_grad():
            for batch_data in test_loader:
                batch = move_to_device(batch_data, self._device)

                loss = self._event_handler.call(Event.TEST_STEP, batch)
                assert loss is not None, "Test step must return a loss value."

                num_samples = get_batch_size(batch)

                progress.update({"loss": loss.item()}, num_samples)

        test_loss = progress.end_phase()
        progress.end_epoch()
        return {"test_loss": test_loss}

    def predict(self, data_loader: DataLoader, verbose: int = 1) -> PredictResult:
        """
        Generate predictions using the model.

        Args:
            data_loader: Data loader for prediction
            verbose: Verbosity level (0=silent, 1=progress bar, 2=simple log)

        Returns:
            PredictResult containing predictions

        Raises:
            ValueError: If no predict step handler is found
        """
        if not self._event_handler.has_handler(Event.PREDICT_STEP):
            msg = (
                "No method decorated with @charge(Event.PREDICT_STEP) found. "
                "Please add a predict step method to your model."
            )
            raise ValueError(msg)

        self._model.eval()
        predictions = []

        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(0)
        progress.start_phase(Phase.PREDICT, total_batches=len(data_loader))

        with torch.no_grad():
            for _, batch_data in enumerate(data_loader):
                batch = move_to_device(batch_data, self._device)

                prediction = self._event_handler.call(Event.PREDICT_STEP, batch)
                if prediction is not None:
                    predictions.append(prediction)

                # Update progress (no loss for predictions)
                progress.update()

        progress.end_phase()
        progress.end_epoch()
        return {"predictions": predictions}
