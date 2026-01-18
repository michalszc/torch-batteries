"""Battery trainer class for torch-batteries."""

from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_batteries.events import Event, EventContext, EventHandler
from torch_batteries.trainer.types import PredictResult, TestResult, TrainResult
from torch_batteries.utils.batch import get_batch_size
from torch_batteries.utils.device import get_device, move_to_device
from torch_batteries.utils.logging import get_logger
from torch_batteries.utils.metrics import calculate_metrics
from torch_batteries.utils.progress import Phase, Progress, ProgressFactory
from torch_batteries.utils.progress.types import (  # noqa: TC001
    ProgressMetrics,
)

logger = get_logger("trainer")


class Battery:
    """A flexible trainer class that uses decorated methods to define training behavior.

    The Battery class discovers methods decorated with `@charge(Event.*)` to automatically
    configure training, validation, testing, and prediction workflows.

    Args:
        model: PyTorch model
        device: PyTorch device. If 'auto', detects available device automatically.
        optimizer: Optimizer for training (optional)
        metrics: Dictionary of metric functions {name: callable(pred, target)}.
                 These metrics are automatically calculated for each batch.
        callbacks: List of callback instances for training events (optional)
    """  # noqa: E501

    __slots__ = (
        "_device",
        "_event_handler",
        "_metrics",
        "_model",
        "_optimizer",
        "_stop_training",
    )

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "auto",
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]]
        | None = None,
        callbacks: list | None = None,
    ):
        self._device = get_device(device)
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._metrics = metrics or {}
        self._event_handler = EventHandler(self._model, callbacks=callbacks)
        self._stop_training = False

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

    @property
    def metrics(
        self,
    ) -> dict[str, Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]]:
        """Get the metrics dictionary."""
        return self._metrics

    @metrics.setter
    def metrics(
        self,
        value: dict[str, Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]]
        | None,
    ) -> None:
        """Set the metrics dictionary."""
        self._metrics = value or {}

    @property
    def stop_training(self) -> bool:
        """Get the stop_training flag."""
        return self._stop_training

    @stop_training.setter
    def stop_training(self, value: bool) -> None:
        """Set the stop_training flag."""
        self._stop_training = value

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

        context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
        }
        self._event_handler.call(Event.BEFORE_TRAIN, context)

        results: TrainResult = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": {},
            "val_metrics": {},
        }

        progress = ProgressFactory.create(verbose=verbose, total_epochs=epochs)

        for epoch in range(epochs):
            if self._stop_training:
                logger.info("Training stopped early at epoch %d.", epoch)
                break

            progress.start_epoch(epoch)

            train_metrics = self._train_epoch(train_loader, progress, epoch)
            results["train_loss"].append(train_metrics["loss"])

            for key, value in train_metrics.items():
                if key != "loss":
                    if key not in results["train_metrics"]:
                        results["train_metrics"][key] = []
                    results["train_metrics"][key].append(value)

            if val_loader:
                before_val_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "epoch": epoch,
                    "train_metrics": train_metrics,
                }
                self._event_handler.call(Event.BEFORE_VALIDATION, before_val_context)

                val_metrics = self._validate_epoch(val_loader, progress, epoch)
                results["val_loss"].append(val_metrics["loss"])

                for key, value in val_metrics.items():
                    if key != "loss":
                        if key not in results["val_metrics"]:
                            results["val_metrics"][key] = []
                        results["val_metrics"][key].append(value)

                after_val_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "epoch": epoch,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }
                self._event_handler.call(Event.AFTER_VALIDATION, after_val_context)

            progress.end_epoch()

        progress.end_training()

        after_train_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": epochs - 1,
            "train_metrics": train_metrics,
        }
        if val_loader and "val_metrics" in locals():
            after_train_context["val_metrics"] = val_metrics
        self._event_handler.call(Event.AFTER_TRAIN, after_train_context)

        return results

    def _train_epoch(
        self, dataloader: DataLoader, progress: Progress, epoch: int
    ) -> dict[str, float]:
        """Run a single training epoch.

        Args:
            dataloader: Training data loader
            progress: Progress tracker instance
            epoch: Current epoch number

        Returns:
            Dictionary with average loss and any additional metrics for the epoch
        """
        # Trigger BEFORE_TRAIN_EPOCH event
        epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": epoch,
        }
        self._event_handler.call(Event.BEFORE_TRAIN_EPOCH, epoch_context)

        self._model.train()

        progress.start_phase(Phase.TRAIN, total_batches=len(dataloader))

        for batch_idx, batch_data in enumerate(dataloader):
            batch = move_to_device(batch_data, self._device)

            before_step_context: EventContext = {
                "battery": self,
                "model": self._model,
                "optimizer": self._optimizer,
                "batch": batch,
                "batch_idx": batch_idx,
                "epoch": epoch,
            }
            self._event_handler.call(Event.BEFORE_TRAIN_STEP, before_step_context)

            # Optimizer is guaranteed to be non-None by train() method
            self._optimizer.zero_grad()  # type: ignore[union-attr]

            step_context: EventContext = {
                "battery": self,
                "model": self._model,
                "optimizer": self._optimizer,
                "batch": batch,
                "batch_idx": batch_idx,
                "epoch": epoch,
            }
            result = self._event_handler.call(Event.TRAIN_STEP, step_context)

            # Handle flexible return: either loss or (loss, metrics_dict)
            step_metrics = {}
            if isinstance(result, tuple):
                loss, step_metrics = result
                assert loss is not None, "Training step must return a loss value."
            else:
                loss = result
                assert loss is not None, "Training step must return a loss value."

            loss.backward()
            self._optimizer.step()  # type: ignore[union-attr]

            init_metrics = {}
            if self._metrics and len(batch) >= 2:
                with torch.no_grad():
                    pred = self._model(batch[0])
                    target = batch[1]
                    init_metrics = calculate_metrics(self._metrics, pred, target)

            batch_metrics = {"loss": loss.item(), **init_metrics, **step_metrics}

            after_step_context: EventContext = {
                "battery": self,
                "model": self._model,
                "optimizer": self._optimizer,
                "batch": batch,
                "batch_idx": batch_idx,
                "epoch": epoch,
                "loss": loss.item(),
                "train_metrics": batch_metrics,
            }
            self._event_handler.call(Event.AFTER_TRAIN_STEP, after_step_context)

            num_samples = get_batch_size(batch)
            progress.update(cast("ProgressMetrics", batch_metrics), num_samples)

        avg_metrics = progress.end_phase()
        train_metrics = (
            avg_metrics if isinstance(avg_metrics, dict) else {"loss": avg_metrics}
        )

        # Trigger AFTER_TRAIN_EPOCH event
        after_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": epoch,
            "train_metrics": train_metrics,
        }
        self._event_handler.call(Event.AFTER_TRAIN_EPOCH, after_epoch_context)

        return train_metrics

    def _validate_epoch(
        self, dataloader: DataLoader, progress: Progress, epoch: int
    ) -> dict[str, float]:
        """Run a single validation epoch.

        Args:
            dataloader: Validation data loader
            progress: Progress tracker instance
            epoch: Current epoch number

        Returns:
            Dictionary with average loss and any additional metrics for the epoch
        """
        if not self._event_handler.has_handler(Event.VALIDATION_STEP):
            msg = (
                "No method decorated with @charge(Event.VALIDATION_STEP) found. "
                "Please add a validation step method to your model."
            )
            raise ValueError(msg)

        # Trigger BEFORE_VALIDATION_EPOCH event
        before_val_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "epoch": epoch,
        }
        self._event_handler.call(
            Event.BEFORE_VALIDATION_EPOCH, before_val_epoch_context
        )

        self._model.eval()

        progress.start_phase(Phase.VALIDATION, total_batches=len(dataloader))

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                batch = move_to_device(batch_data, self._device)

                before_step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": epoch,
                }
                self._event_handler.call(
                    Event.BEFORE_VALIDATION_STEP, before_step_context
                )

                step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": epoch,
                }
                result = self._event_handler.call(Event.VALIDATION_STEP, step_context)

                # Handle flexible return: either loss or (loss, metrics_dict)
                step_metrics = {}
                if isinstance(result, tuple):
                    loss, step_metrics = result
                    assert loss is not None, "Validation step must return a loss value."
                else:
                    loss = result
                    assert loss is not None, "Validation step must return a loss value."

                init_metrics = {}
                if self._metrics and len(batch) >= 2:
                    pred = self._model(batch[0])
                    target = batch[1]
                    init_metrics = calculate_metrics(self._metrics, pred, target)

                batch_metrics = {"loss": loss.item(), **init_metrics, **step_metrics}

                after_step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "val_metrics": batch_metrics,
                }
                self._event_handler.call(
                    Event.AFTER_VALIDATION_STEP, after_step_context
                )

                num_samples = get_batch_size(batch)
                progress.update(cast("ProgressMetrics", batch_metrics), num_samples)

        avg_metrics = progress.end_phase()
        val_metrics = (
            avg_metrics if isinstance(avg_metrics, dict) else {"loss": avg_metrics}
        )

        # Trigger AFTER_VALIDATION_EPOCH event
        after_val_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        self._event_handler.call(Event.AFTER_VALIDATION_EPOCH, after_val_epoch_context)

        return val_metrics

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

        before_test_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
        }
        self._event_handler.call(Event.BEFORE_TEST, before_test_context)

        before_test_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 0,
        }
        self._event_handler.call(Event.BEFORE_TEST_EPOCH, before_test_epoch_context)

        self._model.eval()

        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(0)
        progress.start_phase(Phase.TEST, total_batches=len(test_loader))

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                batch = move_to_device(batch_data, self._device)

                before_step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": 0,
                }
                self._event_handler.call(Event.BEFORE_TEST_STEP, before_step_context)

                step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": 0,
                }
                result = self._event_handler.call(Event.TEST_STEP, step_context)

                # Handle flexible return: either loss or (loss, metrics_dict)
                step_metrics = {}
                if isinstance(result, tuple):
                    loss, step_metrics = result
                    assert loss is not None, "Test step must return a loss value."
                else:
                    loss = result
                    assert loss is not None, "Test step must return a loss value."

                init_metrics = {}
                if self._metrics and len(batch) >= 2:
                    pred = self._model(batch[0])
                    target = batch[1]
                    init_metrics = calculate_metrics(self._metrics, pred, target)

                batch_metrics = {"loss": loss.item(), **init_metrics, **step_metrics}

                after_step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": 0,
                    "loss": loss.item(),
                    "test_metrics": batch_metrics,
                }
                self._event_handler.call(Event.AFTER_TEST_STEP, after_step_context)

                num_samples = get_batch_size(batch)
                progress.update(cast("ProgressMetrics", batch_metrics), num_samples)

        test_metrics = progress.end_phase()
        progress.end_epoch()

        after_test_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 0,
            "loss": test_metrics
            if isinstance(test_metrics, float)
            else test_metrics.get("loss", 0.0),
            "test_metrics": (
                test_metrics
                if isinstance(test_metrics, dict)
                else {"loss": test_metrics}
            ),
        }
        self._event_handler.call(Event.AFTER_TEST_EPOCH, after_test_epoch_context)

        after_test_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "loss": test_metrics
            if isinstance(test_metrics, float)
            else test_metrics.get("loss", 0.0),
            "test_metrics": (
                test_metrics
                if isinstance(test_metrics, dict)
                else {"loss": test_metrics}
            ),
        }
        self._event_handler.call(Event.AFTER_TEST, after_test_context)

        # Format results with test_loss and test_metrics
        if isinstance(test_metrics, dict):
            results: TestResult = {"test_loss": test_metrics["loss"]}

            if len(test_metrics) > 1:  # Has metrics beyond just loss
                results["test_metrics"] = {
                    k: v for k, v in test_metrics.items() if k != "loss"
                }
        else:
            results = {"test_loss": test_metrics}

        return results

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

        before_predict_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
        }
        self._event_handler.call(Event.BEFORE_PREDICT, before_predict_context)

        before_predict_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 0,
        }
        self._event_handler.call(
            Event.BEFORE_PREDICT_EPOCH, before_predict_epoch_context
        )

        self._model.eval()
        predictions = []

        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(0)
        progress.start_phase(Phase.PREDICT, total_batches=len(data_loader))

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                batch = move_to_device(batch_data, self._device)

                before_step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": 0,
                }
                self._event_handler.call(Event.BEFORE_PREDICT_STEP, before_step_context)

                step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": 0,
                }
                prediction = self._event_handler.call(Event.PREDICT_STEP, step_context)
                if prediction is not None:
                    predictions.append(prediction)

                after_step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": 0,
                    "predictions": prediction,
                }
                self._event_handler.call(Event.AFTER_PREDICT_STEP, after_step_context)

                # Update progress (no loss for predictions)
                progress.update()

        progress.end_phase()
        progress.end_epoch()

        after_predict_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 0,
            "predictions": predictions,
        }
        self._event_handler.call(Event.AFTER_PREDICT_EPOCH, after_predict_epoch_context)

        after_predict_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "predictions": predictions,
        }
        self._event_handler.call(Event.AFTER_PREDICT, after_predict_context)

        return {"predictions": predictions}
