"""Battery trainer class for torch-batteries."""

import copy
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_batteries.callbacks.base import Callback
from torch_batteries.events import Event, EventContext, EventHandler, OptimizationStep
from torch_batteries.trainer.context import copy_history_context
from torch_batteries.trainer.types import (
    PredictResult,
    StepOutput,
    TestResult,
    TrainResult,
)
from torch_batteries.utils.batch import get_batch_size
from torch_batteries.utils.device import get_device, move_to_device
from torch_batteries.utils.logging import get_logger
from torch_batteries.utils.metrics import Metric, PhaseMetricManager
from torch_batteries.utils.prediction import concatenate_predictions
from torch_batteries.utils.progress import Phase, Progress, ProgressFactory
from torch_batteries.utils.progress.types import (  # noqa: TC001
    ProgressMetrics,
)

logger = get_logger("trainer")

_CHECKPOINT_SCHEMA_VERSION = 1


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
        "_callbacks",
        "_device",
        "_event_handler",
        "_last_completed_epoch",
        "_metric_manager",
        "_metrics",
        "_model",
        "_optimizer",
        "_optimizer_step_idx",
        "_resume_loaded",
        "_stop_training",
        "_train_results",
    )

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "auto",
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict[str, Metric] | None = None,
        callbacks: list | None = None,
    ):
        self._device = get_device(device)
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._metrics = metrics or {}
        self._metric_manager = PhaseMetricManager(self._metrics)
        callback_list = list(callbacks or [])
        self._callbacks = callback_list
        self._event_handler = EventHandler(self._model, callbacks=callback_list)
        self._stop_training = False
        self._last_completed_epoch = -1
        self._optimizer_step_idx = 0
        self._resume_loaded = False
        self._train_results: TrainResult = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": {},
            "val_metrics": {},
        }
        setup_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "device": self._device,
        }
        self._event_handler.call(Event.SETUP, setup_context)

        logger.debug("Battery initialized on device: %s", self._device)

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
    ) -> dict[str, Metric]:
        """Get the metrics dictionary."""
        return self._metrics

    @metrics.setter
    def metrics(
        self,
        value: dict[str, Metric] | None,
    ) -> None:
        """Set the metrics dictionary."""
        self._metrics = value or {}
        self._metric_manager = PhaseMetricManager(self._metrics)

    @property
    def stop_training(self) -> bool:
        """Get the stop_training flag."""
        return self._stop_training

    @stop_training.setter
    def stop_training(self, value: bool) -> None:
        """Set the stop_training flag."""
        self._stop_training = value

    @staticmethod
    def _validate_loss(loss: Any, phase: str) -> torch.Tensor:
        """Validate and return a scalar loss tensor from a step."""
        if not isinstance(loss, torch.Tensor):
            msg = f"{phase} step loss must be a torch.Tensor."
            raise TypeError(msg)
        if loss.ndim != 0:
            msg = f"{phase} step loss must be a scalar tensor."
            raise ValueError(msg)
        return loss

    @staticmethod
    def _normalize_step_metrics(
        metrics: dict[str, float | torch.Tensor], phase: str
    ) -> dict[str, float]:
        """Convert supported scalar step metrics to Python floats."""
        normalized: dict[str, float] = {}
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    msg = f"Metric '{name}' returned by {phase} step must be scalar."
                    raise ValueError(msg)
                normalized[name] = value.item()
                continue
            try:
                normalized[name] = float(value)
            except (TypeError, ValueError) as error:
                msg = f"Metric '{name}' returned by {phase} step must be numeric."
                raise TypeError(msg) from error
        return normalized

    def _parse_step_result(
        self, result: Any, phase: str
    ) -> tuple[
        torch.Tensor,
        dict[str, float],
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Validate a step result and expose data for configured metrics."""
        if isinstance(result, StepOutput):
            loss = self._validate_loss(result.loss, phase)
            if self._metrics and (result.predictions is None or result.targets is None):
                msg = (
                    f"{phase} step must return StepOutput with predictions and "
                    "targets when Battery metrics are configured."
                )
                raise ValueError(msg)
            manual_metrics = self._normalize_step_metrics(result.metrics, phase)
            return loss, manual_metrics, result.predictions, result.targets

        if self._metrics:
            msg = (
                f"{phase} step must return StepOutput with predictions and targets "
                "when Battery metrics are configured."
            )
            raise ValueError(msg)

        if isinstance(result, tuple):
            if len(result) != 2 or not isinstance(result[1], dict):
                msg = f"{phase} step tuple must be (loss, metrics_dict)."
                raise TypeError(msg)
            loss = self._validate_loss(result[0], phase)
            metrics = self._normalize_step_metrics(result[1], phase)
            return loss, metrics, None, None

        return self._validate_loss(result, phase), {}, None, None

    @staticmethod
    def _validate_loader(dataloader: DataLoader, name: str) -> None:
        """Require a sized, non-empty data loader."""
        try:
            number_of_batches = len(dataloader)
        except TypeError as error:
            msg = f"{name} loader must define its number of batches."
            raise ValueError(msg) from error
        if number_of_batches == 0:
            msg = f"{name} loader must not be empty."
            raise ValueError(msg)

    def _validate_train_inputs(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
    ) -> None:
        """Validate the complete training configuration before events run."""
        if epochs <= 0:
            msg = "epochs must be greater than zero."
            raise ValueError(msg)
        if not self._event_handler.has_handler(Event.TRAIN_STEP):
            msg = (
                "No method decorated with @charge(Event.TRAIN_STEP) found. "
                "Please add a training step method to your model."
            )
            raise ValueError(msg)
        if self._optimizer is None:
            msg = "Optimizer is required for training."
            raise ValueError(msg)

        self._validate_loader(train_loader, "Training")
        if val_loader is None:
            return
        self._validate_loader(val_loader, "Validation")
        if not self._event_handler.has_handler(Event.VALIDATION_STEP):
            msg = (
                "No method decorated with @charge(Event.VALIDATION_STEP) found. "
                "Please add a validation step method to your model."
            )
            raise ValueError(msg)

    def _checkpoint_callbacks(self) -> list[Callback]:
        """Return configured callbacks participating in checkpoint state."""
        return [
            callback for callback in self._callbacks if isinstance(callback, Callback)
        ]

    @staticmethod
    def _callback_identifier(callback: Callback) -> str:
        callback_type = type(callback)
        return f"{callback_type.__module__}.{callback_type.__qualname__}"

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically save complete resumable training state."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks = self._checkpoint_callbacks()
        payload: dict[str, Any] = {
            "__torch_batteries_checkpoint__": _CHECKPOINT_SCHEMA_VERSION,
            "model": self._model.state_dict(),
            "optimizer": (
                self._optimizer.state_dict() if self._optimizer is not None else None
            ),
            "callbacks": [
                {
                    "type": self._callback_identifier(callback),
                    "state": callback.state_dict(),
                }
                for callback in callbacks
            ],
            "metrics": self._metric_manager.state_dict(),
            "epoch": self._last_completed_epoch,
            "optimizer_step_idx": self._optimizer_step_idx,
            "results": copy.deepcopy(self._train_results),
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=checkpoint_path.parent,
                prefix=f".{checkpoint_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
            torch.save(payload, temporary_name)
            Path(temporary_name).replace(checkpoint_path)
        except Exception:
            logger.exception("Failed to save checkpoint at %s.", checkpoint_path)
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            raise
        logger.info(
            "Training checkpoint saved: path=%s, epoch=%d, optimizer_step=%d",
            checkpoint_path,
            self._last_completed_epoch,
            self._optimizer_step_idx,
        )

    @staticmethod
    def _is_raw_model_state(payload: object) -> bool:
        return (
            isinstance(payload, dict)
            and bool(payload)
            and all(isinstance(key, str) for key in payload)
            and all(isinstance(value, torch.Tensor) for value in payload.values())
        )

    @staticmethod
    def _validate_checkpoint_schema(
        payload: object, checkpoint_path: Path
    ) -> dict[str, Any]:
        """Validate and narrow a full-checkpoint payload."""
        if not isinstance(payload, dict):
            logger.error("Checkpoint at %s is not a mapping.", checkpoint_path)
            msg = "Torch-batteries checkpoint structure must be a mapping."
            raise TypeError(msg)

        schema_version = payload.get("__torch_batteries_checkpoint__")
        if schema_version is None:
            logger.error("Unrecognized checkpoint structure at %s.", checkpoint_path)
            msg = "Unrecognized torch-batteries checkpoint structure."
            raise ValueError(msg)
        if schema_version != _CHECKPOINT_SCHEMA_VERSION:
            logger.error(
                "Unsupported checkpoint schema %r at %s; schema %d is required.",
                schema_version,
                checkpoint_path,
                _CHECKPOINT_SCHEMA_VERSION,
            )
            msg = (
                f"Checkpoint schema {schema_version!r} is unsupported; "
                f"schema {_CHECKPOINT_SCHEMA_VERSION} is required."
            )
            raise ValueError(msg)
        return payload

    @staticmethod
    def _move_optimizer_state(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {
                key: Battery._move_optimizer_state(item, device)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [Battery._move_optimizer_state(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(Battery._move_optimizer_state(item, device) for item in value)
        return value

    def load_checkpoint(  # noqa: PLR0915
        self, path: str | Path
    ) -> None:
        """Strictly load full training state or auto-detected raw model weights."""
        checkpoint_path = Path(path)
        try:
            payload = torch.load(
                checkpoint_path,
                map_location=self._device,
                weights_only=True,
            )
        except Exception:
            logger.exception("Failed to read checkpoint at %s.", checkpoint_path)
            raise

        if self._is_raw_model_state(payload):
            logger.warning(
                "Raw model state detected at %s; training state was not restored.",
                checkpoint_path,
            )
            self._model.load_state_dict(payload, strict=True)
            self._resume_loaded = False
            return

        payload = self._validate_checkpoint_schema(payload, checkpoint_path)
        required = {
            "model",
            "optimizer",
            "callbacks",
            "metrics",
            "epoch",
            "optimizer_step_idx",
            "results",
        }
        if not required.issubset(payload):
            missing = sorted(required - set(payload))
            logger.error("Checkpoint is missing required fields: %s", missing)
            msg = f"Training checkpoint is missing fields: {missing}."
            raise ValueError(msg)

        self._model.load_state_dict(payload["model"], strict=True)
        saved_optimizer = payload["optimizer"]
        if saved_optimizer is not None:
            if self._optimizer is None:
                logger.error(
                    "Checkpoint contains optimizer state but Battery does not."
                )
                msg = "An optimizer is required to resume this checkpoint."
                raise ValueError(msg)
            self._optimizer.load_state_dict(saved_optimizer)
            self._optimizer.state = self._move_optimizer_state(
                self._optimizer.state, self._device
            )

        saved_callbacks = payload["callbacks"]
        callbacks = self._checkpoint_callbacks()
        expected_ids = [self._callback_identifier(item) for item in callbacks]
        if not isinstance(saved_callbacks, list):
            logger.error("Checkpoint callback state is not a list.")
            msg = "Invalid callback state in training checkpoint."
            raise TypeError(msg)
        actual_ids = [
            item.get("type") if isinstance(item, dict) else None
            for item in saved_callbacks
        ]
        if actual_ids != expected_ids:
            logger.error(
                "Callback state mismatch: expected=%s, actual=%s",
                expected_ids,
                actual_ids,
            )
            msg = "Configured callbacks do not match checkpoint state."
            raise ValueError(msg)
        for callback, saved in zip(callbacks, saved_callbacks, strict=True):
            callback.load_state_dict(saved["state"])

        metrics_state = payload["metrics"]
        if not isinstance(metrics_state, dict):
            logger.error("Checkpoint metric state is not a dictionary.")
            msg = "Invalid metric state in training checkpoint."
            raise TypeError(msg)
        self._metric_manager.load_state_dict(metrics_state)
        self._last_completed_epoch = int(payload["epoch"])
        self._optimizer_step_idx = int(payload["optimizer_step_idx"])
        results = payload["results"]
        if not isinstance(results, dict):
            logger.error("Checkpoint training results are not a dictionary.")
            msg = "Invalid training history in checkpoint."
            raise TypeError(msg)
        self._train_results = cast("TrainResult", copy.deepcopy(results))
        self._resume_loaded = True
        logger.info(
            "Training checkpoint restored: path=%s, epoch=%d, optimizer_step=%d",
            checkpoint_path,
            self._last_completed_epoch,
            int(payload["optimizer_step_idx"]),
        )

    def train(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 1,
        verbose: int = 1,
        *,
        resume_from: str | Path | None = None,
        resume_epochs_mode: str = "total",
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
        if resume_epochs_mode not in {"total", "additional"}:
            logger.error("Unsupported resume epochs mode: %s", resume_epochs_mode)
            msg = "resume_epochs_mode must be 'total' or 'additional'."
            raise ValueError(msg)
        self._validate_train_inputs(train_loader, val_loader, epochs)
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        resumed = self._resume_loaded
        self._stop_training = False
        if not resumed:
            self._optimizer_step_idx = 0
            self._last_completed_epoch = -1
            self._train_results = {
                "train_loss": [],
                "val_loss": [],
                "train_metrics": {},
                "val_metrics": {},
            }
        logger.info(
            "Training started: epochs=%d, train_batches=%d, validation=%s",
            epochs,
            len(train_loader),
            val_loader is not None,
        )

        context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "resumed": resumed,
        }
        self._event_handler.call(Event.BEFORE_TRAIN, context)

        results = copy.deepcopy(self._train_results)

        progress = ProgressFactory.create(verbose=verbose, total_epochs=epochs)
        train_metrics: dict[str, float] = {}
        val_metrics: dict[str, float] = {}
        last_epoch = self._last_completed_epoch
        start_epoch = self._last_completed_epoch + 1
        stop_epoch = epochs if resume_epochs_mode == "total" else start_epoch + epochs
        if resumed and stop_epoch <= start_epoch:
            logger.error(
                "Resume target does not include new epochs: start=%d, stop=%d",
                start_epoch,
                stop_epoch,
            )
            msg = "Requested resume target does not contain any new epochs."
            raise ValueError(msg)

        for epoch in range(start_epoch, stop_epoch):
            event_epoch = self._event_epoch(epoch)
            if self._stop_training:
                logger.info("Training stopped early at epoch %d.", event_epoch)
                break

            logger.debug("Training epoch started: epoch=%d", event_epoch)
            progress.start_epoch(epoch)

            try:
                train_metrics = self._train_epoch(train_loader, progress, epoch)
            except BaseException:
                progress.abort()
                raise
            results["train_loss"].append(train_metrics["loss"])

            for key, value in train_metrics.items():
                if key != "loss":
                    if key not in results["train_metrics"]:
                        results["train_metrics"][key] = []
                    results["train_metrics"][key].append(value)
            self._last_completed_epoch = epoch
            self._train_results = copy.deepcopy(results)

            after_epoch_context: EventContext = {
                "battery": self,
                "model": self._model,
                "optimizer": self._optimizer,
                "epoch": event_epoch,
                "train_metrics": train_metrics,
                **copy_history_context(results),
            }
            self._event_handler.call(Event.AFTER_TRAIN_EPOCH, after_epoch_context)

            if val_loader:
                logger.debug("Validation phase started: epoch=%d", event_epoch)
                before_val_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "epoch": event_epoch,
                    "train_metrics": train_metrics,
                    **copy_history_context(results),
                }
                self._event_handler.call(Event.BEFORE_VALIDATION, before_val_context)

                try:
                    val_metrics = self._validate_epoch(val_loader, progress, epoch)
                except BaseException:
                    progress.abort()
                    raise
                results["val_loss"].append(val_metrics["loss"])

                for key, value in val_metrics.items():
                    if key != "loss":
                        if key not in results["val_metrics"]:
                            results["val_metrics"][key] = []
                        results["val_metrics"][key].append(value)
                self._train_results = copy.deepcopy(results)

                after_val_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "epoch": event_epoch,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    **copy_history_context(results),
                }
                self._event_handler.call(Event.AFTER_VALIDATION, after_val_context)
                logger.debug(
                    "Validation phase completed: epoch=%d, metrics=%s",
                    event_epoch,
                    val_metrics,
                )

            progress.end_epoch()
            logger.debug(
                "Training epoch completed: epoch=%d, train_metrics=%s",
                event_epoch,
                train_metrics,
            )
            last_epoch = epoch

        progress.end_training()

        after_train_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": self._event_epoch(last_epoch),
            "train_metrics": train_metrics,
            **copy_history_context(results),
        }
        if val_loader and val_metrics:
            after_train_context["val_metrics"] = val_metrics
        self._event_handler.call(Event.AFTER_TRAIN, after_train_context)
        self._train_results = copy.deepcopy(results)
        self._last_completed_epoch = last_epoch
        self._resume_loaded = False
        logger.info(
            "Training completed: completed_epochs=%d, stopped_early=%s",
            len(results["train_loss"]),
            self._stop_training,
        )

        return results

    def _configure_optimization_step(
        self,
        batch: Any,
        batch_idx: int,
        total_batches: int,
        epoch: int,
    ) -> tuple[OptimizationStep, EventContext]:
        """Resolve the optimization plan and its shared batch context."""
        context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "device": self._device,
            "phase": "train",
            "batch": batch,
            "batch_idx": batch_idx,
            "total_batches": total_batches,
            "epoch": epoch,
            "optimizer_step_idx": self._optimizer_step_idx,
        }
        plan = self._event_handler.provide(
            Event.CONFIGURE_TRAIN_STEP,
            context,
            default=OptimizationStep(),
        )
        if not isinstance(plan, OptimizationStep):
            logger.error(
                "Train-step provider returned %s instead of OptimizationStep.",
                type(plan).__name__,
            )
            msg = "CONFIGURE_TRAIN_STEP handler must return an OptimizationStep."
            raise TypeError(msg)
        context["optimization_plan"] = plan
        context["optimizer_step"] = plan.optimizer_step
        return plan, context

    @staticmethod
    def _event_epoch(epoch_index: int) -> int:
        """Translate a private zero-based index into a public epoch number."""
        return epoch_index + 1

    def _run_optimization(
        self,
        loss: torch.Tensor,
        plan: OptimizationStep,
        context: EventContext,
    ) -> None:
        """Run backward and an optional optimizer step through generic events."""
        backward_context: EventContext = {
            **context,
            "loss_tensor": loss,
            "backward_loss": loss / plan.loss_divisor,
        }
        self._event_handler.call(Event.BEFORE_BACKWARD, backward_context)
        backward_loss = backward_context.get("backward_loss")
        if not isinstance(backward_loss, torch.Tensor):
            logger.error("BEFORE_BACKWARD produced a non-tensor backward loss.")
            msg = "BEFORE_BACKWARD must leave backward_loss as a torch.Tensor."
            raise TypeError(msg)
        if not self._event_handler.execute(Event.BACKWARD, backward_context):
            backward_loss.backward()
        self._event_handler.call(Event.AFTER_BACKWARD, backward_context)
        if not plan.optimizer_step:
            return

        self._event_handler.call(Event.BEFORE_GRADIENT_CLIP, backward_context)
        self._event_handler.execute(Event.GRADIENT_CLIP, backward_context)
        self._event_handler.call(Event.BEFORE_OPTIMIZER_STEP, backward_context)
        if not self._event_handler.execute(Event.OPTIMIZER_STEP, backward_context):
            self._optimizer.step()  # type: ignore[union-attr]
        self._optimizer_step_idx += 1
        backward_context["optimizer_step_idx"] = self._optimizer_step_idx
        self._event_handler.call(Event.AFTER_OPTIMIZER_STEP, backward_context)

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
        event_epoch = self._event_epoch(epoch)

        # Trigger BEFORE_TRAIN_EPOCH event
        epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": event_epoch,
        }
        self._event_handler.call(Event.BEFORE_TRAIN_EPOCH, epoch_context)

        self._model.train()

        progress.start_phase(Phase.TRAIN, total_batches=len(dataloader))
        self._metric_manager.reset()
        manual_metric_names: set[str] = set()
        logger.debug("Training phase started: epoch=%d", event_epoch)

        total_batches = len(dataloader)
        for batch_idx, batch_data in enumerate(dataloader):
            batch = move_to_device(batch_data, self._device)

            optimization_plan, before_step_context = self._configure_optimization_step(
                batch,
                batch_idx,
                total_batches,
                event_epoch,
            )

            if optimization_plan.zero_grad:
                # Optimizer is guaranteed to be non-None by train() method
                self._optimizer.zero_grad()  # type: ignore[union-attr]
                logger.debug(
                    "Gradients cleared: epoch=%d, batch=%d",
                    event_epoch,
                    batch_idx,
                )
            self._event_handler.call(Event.BEFORE_TRAIN_STEP, before_step_context)

            step_context: EventContext = {
                **before_step_context,
            }
            with self._event_handler.execution_context(
                Event.STEP_EXECUTION_CONTEXT, step_context
            ):
                result = self._event_handler.call(Event.TRAIN_STEP, step_context)

            loss, step_metrics, predictions, targets = self._parse_step_result(
                result, "Training"
            )
            automatic_metrics = (
                self._metric_manager.update(predictions, targets)
                if predictions is not None and targets is not None
                else {}
            )
            manual_metric_names.update(step_metrics)

            self._run_optimization(loss, optimization_plan, before_step_context)
            optimizer_step = optimization_plan.optimizer_step

            batch_metrics = {
                "loss": loss.item(),
                **automatic_metrics,
                **step_metrics,
            }
            logger.debug(
                "Training step completed: epoch=%d, batch=%d, metrics=%s",
                event_epoch,
                batch_idx,
                batch_metrics,
            )

            after_step_context: EventContext = {
                "battery": self,
                "model": self._model,
                "optimizer": self._optimizer,
                "batch": batch,
                "batch_idx": batch_idx,
                "epoch": event_epoch,
                "loss": loss.item(),
                "train_loss": loss.item(),
                "train_metrics": batch_metrics,
                "optimizer_step": optimizer_step,
                "optimizer_step_idx": self._optimizer_step_idx,
                "optimization_plan": optimization_plan,
            }
            self._event_handler.call(Event.AFTER_TRAIN_STEP, after_step_context)

            num_samples = get_batch_size(batch)
            progress.update(cast("ProgressMetrics", batch_metrics), num_samples)

        avg_metrics = progress.end_phase()
        train_metrics = (
            avg_metrics if isinstance(avg_metrics, dict) else {"loss": avg_metrics}
        )
        train_metrics.update(
            {
                name: value
                for name, value in self._metric_manager.compute().items()
                if name not in manual_metric_names
            }
        )
        logger.debug(
            "Training phase completed: epoch=%d, metrics=%s",
            event_epoch,
            train_metrics,
        )
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

        event_epoch = self._event_epoch(epoch)

        # Trigger BEFORE_VALIDATION_EPOCH event
        before_val_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "epoch": event_epoch,
        }
        self._event_handler.call(
            Event.BEFORE_VALIDATION_EPOCH, before_val_epoch_context
        )

        self._model.eval()

        progress.start_phase(Phase.VALIDATION, total_batches=len(dataloader))
        self._metric_manager.reset()
        manual_metric_names: set[str] = set()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                batch = move_to_device(batch_data, self._device)

                before_step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "device": self._device,
                    "phase": "validation",
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": event_epoch,
                }
                self._event_handler.call(
                    Event.BEFORE_VALIDATION_STEP, before_step_context
                )

                step_context: EventContext = {
                    **before_step_context,
                }
                with self._event_handler.execution_context(
                    Event.STEP_EXECUTION_CONTEXT, step_context
                ):
                    result = self._event_handler.call(
                        Event.VALIDATION_STEP, step_context
                    )

                loss, step_metrics, predictions, targets = self._parse_step_result(
                    result, "Validation"
                )
                automatic_metrics = (
                    self._metric_manager.update(predictions, targets)
                    if predictions is not None and targets is not None
                    else {}
                )
                manual_metric_names.update(step_metrics)
                batch_metrics = {
                    "loss": loss.item(),
                    **automatic_metrics,
                    **step_metrics,
                }
                logger.debug(
                    "Validation step completed: epoch=%d, batch=%d, metrics=%s",
                    event_epoch,
                    batch_idx,
                    batch_metrics,
                )

                after_step_context: EventContext = {
                    "battery": self,
                    "model": self._model,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": event_epoch,
                    "loss": loss.item(),
                    "val_loss": loss.item(),
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
        val_metrics.update(
            {
                name: value
                for name, value in self._metric_manager.compute().items()
                if name not in manual_metric_names
            }
        )

        # Trigger AFTER_VALIDATION_EPOCH event
        after_val_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "epoch": event_epoch,
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

        self._validate_loader(test_loader, "Test")
        logger.info("Testing started: batches=%d", len(test_loader))

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
            "epoch": 1,
        }
        self._event_handler.call(Event.BEFORE_TEST_EPOCH, before_test_epoch_context)

        self._model.eval()

        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(0)
        progress.start_phase(Phase.TEST, total_batches=len(test_loader))
        self._metric_manager.reset()
        manual_metric_names: set[str] = set()
        logger.debug("Test phase started: epoch=1")

        try:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    self._test_batch(
                        batch_data,
                        batch_idx,
                        progress,
                        manual_metric_names,
                    )
        except BaseException:
            progress.abort()
            raise

        test_metrics = progress.end_phase()
        progress.end_epoch()
        test_loss = (
            test_metrics
            if isinstance(test_metrics, float)
            else test_metrics.get("loss", 0.0)
        )
        test_metrics_context = (
            test_metrics if isinstance(test_metrics, dict) else {"loss": test_metrics}
        )
        test_metrics_context.update(
            {
                name: value
                for name, value in self._metric_manager.compute().items()
                if name not in manual_metric_names
            }
        )

        after_test_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 1,
            "loss": test_loss,
            "test_loss": test_loss,
            "test_metrics": test_metrics_context,
        }
        self._event_handler.call(Event.AFTER_TEST_EPOCH, after_test_epoch_context)

        after_test_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "loss": test_loss,
            "test_loss": test_loss,
            "test_metrics": test_metrics_context,
        }
        self._event_handler.call(Event.AFTER_TEST, after_test_context)
        logger.debug("Test phase completed: epoch=1, metrics=%s", test_metrics_context)

        results: TestResult = {"test_loss": test_metrics_context["loss"]}
        if len(test_metrics_context) > 1:
            results["test_metrics"] = {
                key: value
                for key, value in test_metrics_context.items()
                if key != "loss"
            }

        logger.info("Testing completed")
        return results

    def _test_batch(
        self,
        batch_data: Any,
        batch_idx: int,
        progress: Progress,
        manual_metric_names: set[str],
    ) -> None:
        """Process one test batch."""
        batch = move_to_device(batch_data, self._device)

        before_step_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "device": self._device,
            "phase": "test",
            "batch": batch,
            "batch_idx": batch_idx,
            "epoch": 1,
        }
        self._event_handler.call(Event.BEFORE_TEST_STEP, before_step_context)

        step_context: EventContext = {
            **before_step_context,
        }
        with self._event_handler.execution_context(
            Event.STEP_EXECUTION_CONTEXT, step_context
        ):
            result = self._event_handler.call(Event.TEST_STEP, step_context)

        loss, step_metrics, predictions, targets = self._parse_step_result(
            result, "Test"
        )
        automatic_metrics = (
            self._metric_manager.update(predictions, targets)
            if predictions is not None and targets is not None
            else {}
        )
        manual_metric_names.update(step_metrics)
        batch_metrics = {
            "loss": loss.item(),
            **automatic_metrics,
            **step_metrics,
        }
        logger.debug(
            "Test step completed: epoch=1, batch=%d, metrics=%s",
            batch_idx,
            batch_metrics,
        )

        after_step_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "batch": batch,
            "batch_idx": batch_idx,
            "epoch": 1,
            "loss": loss.item(),
            "test_loss": loss.item(),
            "test_metrics": batch_metrics,
        }
        self._event_handler.call(Event.AFTER_TEST_STEP, after_step_context)

        num_samples = get_batch_size(batch)
        progress.update(cast("ProgressMetrics", batch_metrics), num_samples)

    def predict(
        self,
        data_loader: DataLoader,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
        concatenate: bool = False,
    ) -> PredictResult:
        """
        Generate predictions using the model.

        Args:
            data_loader: Data loader for prediction
            verbose: Verbosity level (0=silent, 1=progress bar, 2=simple log)
            move_to_cpu: Recursively detach tensor outputs and move them to CPU.
                This is useful when predictions should not retain accelerator memory.
            concatenate: Recursively concatenate matching tensor outputs along their
                first dimension. Nested dictionaries, tuples, named tuples, and lists
                retain their structure.

        Returns:
            PredictResult containing predictions

        Raises:
            ValueError: If no predict step handler is found
        """
        if not self._event_handler.has_handler(Event.PREDICT_STEP):
            logger.error("Prediction requires a predict step handler.")
            msg = (
                "No method decorated with @charge(Event.PREDICT_STEP) found. "
                "Please add a predict step method to your model."
            )
            raise ValueError(msg)

        self._validate_loader(data_loader, "Prediction")
        logger.info("Prediction started: batches=%d", len(data_loader))
        logger.debug(
            "Prediction output options selected: move_to_cpu=%s, concatenate=%s",
            move_to_cpu,
            concatenate,
        )

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
            "epoch": 1,
        }
        self._event_handler.call(
            Event.BEFORE_PREDICT_EPOCH, before_predict_epoch_context
        )

        self._model.eval()
        predictions: list[Any] = []

        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(0)
        progress.start_phase(Phase.PREDICT, total_batches=len(data_loader))
        logger.debug("Prediction phase started: epoch=1")

        try:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(data_loader):
                    prediction = self._predict_batch(
                        batch_data,
                        batch_idx,
                        progress,
                        move_to_cpu=move_to_cpu,
                    )
                    if prediction is not None:
                        predictions.append(prediction)
        except BaseException:
            progress.abort()
            raise

        progress.end_phase()
        progress.end_epoch()

        prediction_output = (
            concatenate_predictions(predictions) if concatenate else predictions
        )
        after_predict_epoch_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 1,
            "predictions": prediction_output,
            "prediction_batches": len(predictions),
        }
        self._event_handler.call(Event.AFTER_PREDICT_EPOCH, after_predict_epoch_context)

        after_predict_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "predictions": prediction_output,
            "prediction_batches": len(predictions),
        }
        self._event_handler.call(Event.AFTER_PREDICT, after_predict_context)
        logger.debug(
            "Prediction phase completed: epoch=1, outputs=%d", len(predictions)
        )
        logger.info("Prediction completed: outputs=%d", len(predictions))

        logger.debug(
            "Prediction output options applied: move_to_cpu=%s, concatenate=%s",
            move_to_cpu,
            concatenate,
        )
        return {"predictions": prediction_output}

    def predict_iter(
        self,
        data_loader: DataLoader,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
    ) -> Iterator[Any]:
        """Yield prediction batches without retaining the complete result.

        Args:
            data_loader: Data loader for prediction.
            verbose: Verbosity level (0=silent, 1=progress bar, 2=simple log).
            move_to_cpu: Recursively detach tensor outputs and move them to CPU
                before yielding them.

        Yields:
            One prediction-step output at a time. Iteration must finish for the
            ``AFTER_PREDICT_EPOCH`` and ``AFTER_PREDICT`` events to run.
        """
        if not self._event_handler.has_handler(Event.PREDICT_STEP):
            logger.error("Streaming prediction requires a predict step handler.")
            msg = (
                "No method decorated with @charge(Event.PREDICT_STEP) found. "
                "Please add a predict step method to your model."
            )
            raise ValueError(msg)
        self._validate_loader(data_loader, "Prediction")
        logger.info(
            "Streaming prediction started: batches=%d, move_to_cpu=%s",
            len(data_loader),
            move_to_cpu,
        )
        yield from self._prediction_iterator(
            data_loader,
            verbose,
            move_to_cpu=move_to_cpu,
        )

    def _prediction_iterator(
        self,
        data_loader: DataLoader,
        verbose: int,
        *,
        move_to_cpu: bool,
    ) -> Iterator[Any]:
        """Run lazy prediction lifecycle and yield each non-None output."""
        before_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
        }
        self._event_handler.call(Event.BEFORE_PREDICT, before_context)
        self._event_handler.call(
            Event.BEFORE_PREDICT_EPOCH,
            {**before_context, "epoch": 1},
        )
        self._model.eval()
        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(0)
        progress.start_phase(Phase.PREDICT, total_batches=len(data_loader))
        processed_batches = 0
        completed = False
        try:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(data_loader):
                    prediction = self._predict_batch(
                        batch_data,
                        batch_idx,
                        progress,
                        move_to_cpu=move_to_cpu,
                    )
                    processed_batches += 1
                    if prediction is not None:
                        yield prediction
            progress.end_phase()
            progress.end_epoch()
            completion_context: EventContext = {
                "battery": self,
                "model": self._model,
                "optimizer": self._optimizer,
                "epoch": 1,
                "prediction_batches": processed_batches,
            }
            self._event_handler.call(Event.AFTER_PREDICT_EPOCH, completion_context)
            completion_context.pop("epoch")
            self._event_handler.call(Event.AFTER_PREDICT, completion_context)
            completed = True
            logger.info("Streaming prediction completed: batches=%d", processed_batches)
        finally:
            if not completed:
                progress.abort()
                logger.warning(
                    "Streaming prediction aborted after %d batches.",
                    processed_batches,
                )

    def _predict_batch(
        self,
        batch_data: Any,
        batch_idx: int,
        progress: Progress,
        *,
        move_to_cpu: bool = False,
    ) -> Any:
        """Process one prediction batch."""
        batch = move_to_device(batch_data, self._device)

        before_step_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "device": self._device,
            "phase": "predict",
            "batch": batch,
            "batch_idx": batch_idx,
            "epoch": 1,
        }
        self._event_handler.call(Event.BEFORE_PREDICT_STEP, before_step_context)

        step_context: EventContext = {
            **before_step_context,
        }
        with self._event_handler.execution_context(
            Event.STEP_EXECUTION_CONTEXT, step_context
        ):
            prediction = self._event_handler.call(Event.PREDICT_STEP, step_context)
        if move_to_cpu:
            prediction = move_to_device(
                prediction,
                torch.device("cpu"),
                detach=True,
            )
        logger.debug(
            "Prediction step completed: epoch=1, batch=%d, output_type=%s",
            batch_idx,
            type(prediction).__name__,
        )
        after_step_context: EventContext = {
            "battery": self,
            "model": self._model,
            "optimizer": self._optimizer,
            "batch": batch,
            "batch_idx": batch_idx,
            "epoch": 1,
            "predictions": prediction,
        }
        self._event_handler.call(Event.AFTER_PREDICT_STEP, after_step_context)

        progress.update()
        return prediction
