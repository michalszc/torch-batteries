"""Battery trainer class for torch-batteries."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, overload

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_batteries.data import DataPack, ResolvedData
from torch_batteries.data.handler import DataPackHandler
from torch_batteries.data.types import DataStage
from torch_batteries.events import Event, EventContext, EventHandler
from torch_batteries.trainer.types import (
    FitResult,
    PredictResult,
    StepOutput,
    TestResult,
    TrainResult,
    ValidationResult,
)
from torch_batteries.utils.device import get_device
from torch_batteries.utils.logging import get_logger
from torch_batteries.utils.metrics import Metric, PhaseMetricManager

from ._checkpoint import CheckpointMixin
from ._evaluation import EvaluationMixin
from ._prediction import PredictionMixin
from ._training import TrainingMixin

logger = get_logger("trainer.core")


class Battery(CheckpointMixin, TrainingMixin, EvaluationMixin, PredictionMixin):
    """Run event-driven training, evaluation, and prediction for a PyTorch model.

    ``Battery`` discovers model methods decorated with :func:`~torch_batteries.charge`
    and dispatches lifecycle events to the model and configured callbacks. It moves
    the model and batches to the selected device, aggregates losses and metrics, and
    can save complete resumable training state.

    Args:
        model: Model containing the charged step methods. It is moved to ``device``.
        device: Explicit PyTorch device or ``"auto"``. Automatic selection prefers
            CUDA, then MPS, then CPU.
        optimizer: Optimizer used by :meth:`train`. It is optional for testing and
            prediction.
        metrics: Named callable or stateful metrics. When metrics are configured,
            train, validation, and test steps must return :class:`StepOutput` with
            predictions and targets.
        callbacks: Ordered callback objects. Callback order is significant for
            provider-style optimization events.
        data_pack: Optional event-driven dataset and DataLoader configuration. When
            attached, workflow loaders may be omitted.

    Note:
        Epoch values exposed through event contexts are one-based. Prediction output
        stays on its current device unless ``move_to_cpu=True`` is requested.
    """

    __slots__ = (
        "_callbacks",
        "_data_pack",
        "_data_pack_handler",
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

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        device: str | torch.device = "auto",
        optimizer: torch.optim.Optimizer | None = None,
        metrics: dict[str, Metric] | None = None,
        callbacks: list | None = None,
        *,
        data_pack: DataPack | None = None,
    ):
        self._device = get_device(device)
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._metrics = metrics or {}
        self._metric_manager = PhaseMetricManager(self._metrics)
        callback_list = list(callbacks or [])
        self._callbacks = callback_list
        self._event_handler = EventHandler(self._model, callbacks=callback_list)
        self._data_pack = data_pack
        self._data_pack_handler = (
            DataPackHandler(data_pack) if data_pack is not None else None
        )
        self._stop_training = False
        self._last_completed_epoch = 0
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
    def data_pack(self) -> DataPack | None:
        """Get the event-driven data configuration attached to this Battery."""
        return self._data_pack

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

    def save_checkpoint(self, path: str | Path) -> None:
        """Save complete resumable training state atomically.

        Args:
            path: Destination checkpoint path. Parent directories are created.

        Raises:
            OSError: If the destination cannot be created or replaced.
            Exception: If PyTorch serialization fails.
        """
        CheckpointMixin.save_checkpoint(self, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore a full checkpoint or raw model state.

        Args:
            path: Trusted checkpoint or model-state path.

        Raises:
            ValueError: If saved state is incompatible with this Battery.
            TypeError: If the checkpoint structure is invalid.
            RuntimeError: If strict PyTorch state restoration fails.
        """
        CheckpointMixin.load_checkpoint(self, path)

    def train(  # noqa: PLR0913
        self,
        train_loader: DataLoader | None = None,
        # Deprecated compatibility parameter; use fit(..., val_loader=...).
        val_loader: DataLoader | None = None,
        epochs: int = 1,
        verbose: int = 1,
        *,
        resume_from: str | Path | None = None,
        resume_epochs_mode: str = "total",
    ) -> TrainResult:
        """Train with explicit loaders or the attached DataPack.

        Validation through this method is deprecated. Use ``fit`` for combined
        training and validation. Calls without validation data do not warn.

        Args:
            train_loader: Optional sized, non-empty training loader.
            val_loader: Deprecated. Optional validation loader for direct-loader
                compatibility. Use ``fit`` for validated training.
            epochs: Positive epoch count or resume target.
            verbose: ``0`` for silent, ``1`` for bars, or ``2`` for summaries.
            resume_from: Optional full checkpoint restored before data setup.
            resume_epochs_mode: ``"total"`` or ``"additional"``.

        Returns:
            Per-epoch loss and named metric histories.
        """
        return TrainingMixin.train(
            self,
            train_loader,
            val_loader,
            epochs,
            verbose,
            resume_from=resume_from,
            resume_epochs_mode=resume_epochs_mode,
        )

    def fit(  # noqa: PLR0913
        self,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        epochs: int = 1,
        verbose: int = 1,
        *,
        resume_from: str | Path | None = None,
        resume_epochs_mode: str = "total",
    ) -> FitResult:
        """Train with optional per-epoch validation.

        Args:
            train_loader: Optional sized, non-empty training loader.
            val_loader: Optional validation loader for direct-loader mode.
            epochs: Positive epoch count or resume target.
            verbose: ``0`` for silent, ``1`` for bars, or ``2`` for summaries.
            resume_from: Optional full checkpoint restored before data setup.
            resume_epochs_mode: ``"total"`` or ``"additional"``.

        Returns:
            Per-epoch training histories and optional validation histories. Validation
            histories are empty when validation data is unavailable.
        """
        return TrainingMixin.fit(
            self,
            train_loader,
            val_loader,
            epochs,
            verbose,
            resume_from=resume_from,
            resume_epochs_mode=resume_epochs_mode,
        )

    def validate(
        self,
        val_loader: DataLoader | None = None,
        verbose: int = 1,
    ) -> ValidationResult:
        """Run one standalone validation pass.

        Args:
            val_loader: Optional sized, non-empty validation loader. When omitted,
                the attached DataPack must provide validation data.
            verbose: ``0`` for silent, ``1`` for a bar, or ``2`` for a summary.

        Returns:
            Aggregate validation loss and optional named validation metrics.
        """
        return EvaluationMixin._validate(self, val_loader, verbose)  # noqa: SLF001

    @overload
    def test(
        self,
        test_loader: DataLoader,
        verbose: int = 1,
        *,
        dataset: None = None,
    ) -> TestResult: ...

    @overload
    def test(
        self,
        test_loader: None = None,
        verbose: int = 1,
        *,
        dataset: str,
    ) -> TestResult: ...

    @overload
    def test(
        self,
        test_loader: DataLoader | None = None,
        verbose: int = 1,
        *,
        dataset: None = None,
    ) -> TestResult | dict[str, TestResult]: ...

    def test(
        self,
        test_loader: DataLoader | None = None,
        verbose: int = 1,
        *,
        dataset: str | None = None,
    ) -> TestResult | dict[str, TestResult]:
        """Evaluate an explicit or DataPack-provided test dataset.

        Args:
            test_loader: Optional sized, non-empty test loader.
            verbose: ``0`` for silent, ``1`` for a bar, or ``2`` for a summary.
            dataset: Optional DataPack test dataset name.

        Returns:
            One result or a mapping of named DataPack results.
        """
        return EvaluationMixin._test(self, test_loader, verbose, dataset=dataset)  # noqa: SLF001

    @overload
    def predict(
        self,
        data_loader: DataLoader,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
        concatenate: bool = False,
        dataset: None = None,
    ) -> PredictResult: ...

    @overload
    def predict(
        self,
        data_loader: None = None,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
        concatenate: bool = False,
        dataset: str,
    ) -> PredictResult: ...

    @overload
    def predict(
        self,
        data_loader: DataLoader | None = None,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
        concatenate: bool = False,
        dataset: None = None,
    ) -> PredictResult | dict[str, PredictResult]: ...

    def predict(
        self,
        data_loader: DataLoader | None = None,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
        concatenate: bool = False,
        dataset: str | None = None,
    ) -> PredictResult | dict[str, PredictResult]:
        """Collect predictions from an explicit or DataPack loader.

        Args:
            data_loader: Optional sized, non-empty prediction loader.
            verbose: ``0`` for silent, ``1`` for a bar, or ``2`` for a summary.
            move_to_cpu: Detach tensor outputs and move them to CPU.
            concatenate: Concatenate compatible batch output structures.
            dataset: Optional DataPack prediction dataset name.

        Returns:
            One prediction result or a mapping of named results.
        """
        return PredictionMixin._predict(  # noqa: SLF001
            self,
            data_loader,
            verbose,
            move_to_cpu=move_to_cpu,
            concatenate=concatenate,
            dataset=dataset,
        )

    def predict_iter(
        self,
        data_loader: DataLoader | None = None,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
        dataset: str | None = None,
    ) -> Generator[Any]:
        """Yield predictions from an explicit or DataPack loader.

        Args:
            data_loader: Optional sized, non-empty prediction loader.
            verbose: ``0`` for silent, ``1`` for a bar, or ``2`` for a summary.
            move_to_cpu: Detach tensor outputs and move them to CPU.
            dataset: Optional DataPack prediction dataset name.

        Yields:
            One prediction-step output at a time.
        """
        yield from PredictionMixin.predict_iter(
            self,
            data_loader,
            verbose,
            move_to_cpu=move_to_cpu,
            dataset=dataset,
        )

    @contextmanager
    def _data_workflow(
        self,
        stage: DataStage,
        *,
        dataset_name: str | None = None,
    ) -> Generator[ResolvedData]:
        """Resolve implicit loaders and guarantee DataPack teardown."""
        if self._data_pack_handler is None or self._data_pack is None:
            msg = (
                "No DataLoader was provided and Battery has no DataPack. "
                "Pass a loader or configure Battery(data_pack=...)."
            )
            raise ValueError(msg)

        with self._data_pack_handler.resolve(
            stage,
            device=self._device,
            battery=self,
            dataset_name=dataset_name,
        ) as resolved:
            yield resolved

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
    def _validate_loader(dataloader: object, name: str) -> None:
        """Require a sized, non-empty data loader."""
        if not isinstance(dataloader, DataLoader):
            msg = f"{name} loader must be a torch.utils.data.DataLoader."
            raise TypeError(msg)
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
    ) -> None:
        """Validate the complete training configuration before events run."""
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
