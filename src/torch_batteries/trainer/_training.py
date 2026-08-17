"""Training workflows for ``torch_batteries.Battery``."""

import copy
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader

from torch_batteries.events import Event, EventContext, OptimizationStep
from torch_batteries.trainer.context import copy_history_context
from torch_batteries.trainer.types import FitResult, TrainResult
from torch_batteries.utils.batch import get_batch_size
from torch_batteries.utils.device import move_to_device
from torch_batteries.utils.logging import get_logger
from torch_batteries.utils.progress import Phase, Progress, ProgressFactory

from ._state import BatteryStateMixin, as_battery

if TYPE_CHECKING:
    from torch_batteries.utils.progress.types import ProgressMetrics

logger = get_logger("trainer._training")


class TrainingMixin(BatteryStateMixin):
    """Implement training and validation loops."""

    __slots__ = ()

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
        """Fit the model with optional per-epoch validation.

        Args:
            train_loader: Optional sized, non-empty training loader.
            val_loader: Optional validation loader for direct-loader mode.
            epochs: Positive epoch count or resume target.
            verbose: ``0`` for silent, ``1`` for bars, or ``2`` for summaries.
            resume_from: Optional full checkpoint restored before data setup.
            resume_epochs_mode: ``"total"`` or ``"additional"``.

        Returns:
            Per-epoch training and optional validation histories.
        """
        return self._run_training_workflow(
            train_loader,
            val_loader,
            epochs,
            verbose,
            resume_from=resume_from,
            resume_epochs_mode=resume_epochs_mode,
            warn_for_validation=False,
        )

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
        """Train the model for one or more epochs.

        Passing ``train_loader`` selects direct-loader mode. When it is omitted, the
        attached DataPack supplies train and optional validation loaders. A checkpoint
        passed through ``resume_from`` is restored before DataPack setup.

        Args:
            train_loader: Optional sized, non-empty training loader.
            val_loader: Deprecated. Optional validation loader used only with an
                explicit train loader. Use :meth:`fit` for validated training.
                Implicit validation compatibility comes from the DataPack.
            epochs: Positive epoch count or resume target.
            verbose: ``0`` for silent, ``1`` for progress bars, or ``2`` for summaries.
            resume_from: Optional full checkpoint restored before data resolution.
            resume_epochs_mode: ``"total"`` treats ``epochs`` as the final target;
                ``"additional"`` runs that many new epochs.

        Returns:
            Per-epoch loss histories and named metric histories.

        Raises:
            ValueError: If loaders, DataPack datasets, handlers, optimizer, resume
                mode, or checkpoint state are incompatible.
        """
        return self._run_training_workflow(
            train_loader,
            val_loader,
            epochs,
            verbose,
            resume_from=resume_from,
            resume_epochs_mode=resume_epochs_mode,
            warn_for_validation=True,
        )

    def _run_training_workflow(  # noqa: PLR0913
        self,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        epochs: int,
        verbose: int,
        *,
        resume_from: str | Path | None,
        resume_epochs_mode: str,
        warn_for_validation: bool,
    ) -> FitResult:
        """Resolve loaders and run the shared training engine."""
        if epochs <= 0:
            msg = "epochs must be greater than zero."
            raise ValueError(msg)
        if resume_epochs_mode not in {"total", "additional"}:
            logger.error("Unsupported resume epochs mode: %s", resume_epochs_mode)
            msg = "resume_epochs_mode must be 'total' or 'additional'."
            raise ValueError(msg)
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        if train_loader is not None:
            return self._train_with_loaders(
                train_loader,
                val_loader,
                epochs,
                verbose,
                resume_epochs_mode=resume_epochs_mode,
                warn_for_validation=warn_for_validation,
            )
        if val_loader is not None:
            msg = (
                "An explicit validation loader cannot be combined with an implicit "
                "DataPack training loader."
            )
            raise ValueError(msg)
        with self._data_workflow("fit") as workflow:
            train_loaders = workflow.loaders.loaders_for_phase("train")
            validation_loaders = workflow.loaders.loaders_for_phase("validation")
            return self._train_with_loaders(
                next(iter(train_loaders.values())),
                next(iter(validation_loaders.values()), None),
                epochs,
                verbose,
                resume_epochs_mode=resume_epochs_mode,
                warn_for_validation=warn_for_validation,
            )

    def _train_with_loaders(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 1,
        verbose: int = 1,
        *,
        resume_epochs_mode: str = "total",
        warn_for_validation: bool = False,
    ) -> FitResult:
        """Train the model for one or more epochs.

        A fresh call resets history and optimizer-step counters. A checkpoint loaded
        before this method is called continues its stored history. With
        ``resume_epochs_mode="total"``, ``epochs`` is the final epoch target; with
        ``"additional"``, it is the number of new epochs to run.

        Args:
            train_loader: Sized, non-empty training loader.
            val_loader: Optional sized, non-empty validation loader. Supplying one
                requires a method charged for ``Event.VALIDATION_STEP``.
            epochs: Positive epoch count or resume target, depending on
                ``resume_epochs_mode``.
            verbose: ``0`` for silent, ``1`` for progress bars, or ``2`` for summaries.
            resume_epochs_mode: ``"total"`` or ``"additional"``.
            warn_for_validation: Whether to warn when the compatibility validation
                behavior of :meth:`train` actually runs.

        Returns:
            Per-epoch loss histories and named metric histories. Validation entries
            remain empty when no validation loader is supplied.

        Raises:
            ValueError: If inputs, handlers, resume mode, or checkpoint state are
                incompatible.
            TypeError: If a step result has an unsupported structure.
        """
        self._validate_train_inputs(train_loader, val_loader)
        resumed = self._resume_loaded
        self._stop_training = False
        if not resumed:
            self._optimizer_step_idx = 0
            self._last_completed_epoch = 0
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
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "resumed": resumed,
        }
        self._event_handler.call(Event.BEFORE_TRAIN, context)

        results = copy.deepcopy(self._train_results)

        train_metrics: dict[str, float] = {}
        val_metrics: dict[str, float] = {}
        last_epoch = self._last_completed_epoch
        start_epoch = self._last_completed_epoch + 1
        stop_epoch = (
            epochs + 1 if resume_epochs_mode == "total" else start_epoch + epochs
        )
        if resumed and stop_epoch <= start_epoch:
            logger.error(
                "Resume target does not include new epochs: start=%d, stop=%d",
                start_epoch,
                stop_epoch,
            )
            msg = "Requested resume target does not contain any new epochs."
            raise ValueError(msg)

        progress = ProgressFactory.create(
            verbose=verbose,
            total_epochs=stop_epoch - 1,
        )
        for epoch in range(start_epoch, stop_epoch):
            if self._stop_training:
                logger.info("Training stopped early at epoch %d.", epoch)
                break

            logger.debug("Training epoch started: epoch=%d", epoch)
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
                "battery": as_battery(self),
                "model": self._model,
                "optimizer": self._optimizer,
                "epoch": epoch,
                "train_metrics": train_metrics,
                **copy_history_context(results),
            }
            self._event_handler.call(Event.AFTER_TRAIN_EPOCH, after_epoch_context)

            if val_loader:
                if warn_for_validation:
                    # A future version will make train() training-only and will no
                    # longer run validation.
                    warning_message = (
                        "Validation through Battery.train() is deprecated and will "
                        "be removed from train() in a future version. Use "
                        "Battery.fit() for combined training and validation."
                    )
                    logger.warning(warning_message)
                    warnings.warn(
                        warning_message,
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    warn_for_validation = False
                logger.debug("Validation phase started: epoch=%d", epoch)
                before_val_context: EventContext = {
                    "battery": as_battery(self),
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "epoch": epoch,
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
                    "battery": as_battery(self),
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "epoch": epoch,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    **copy_history_context(results),
                }
                self._event_handler.call(Event.AFTER_VALIDATION, after_val_context)
                logger.debug(
                    "Validation phase completed: epoch=%d, metrics=%s",
                    epoch,
                    val_metrics,
                )

            progress.end_epoch()
            logger.debug(
                "Training epoch completed: epoch=%d, train_metrics=%s",
                epoch,
                train_metrics,
            )
            last_epoch = epoch

        progress.end_training()

        after_train_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": last_epoch,
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
            "battery": as_battery(self),
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
        # Trigger BEFORE_TRAIN_EPOCH event
        epoch_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": epoch,
        }
        self._event_handler.call(Event.BEFORE_TRAIN_EPOCH, epoch_context)

        self._model.train()

        progress.start_phase(Phase.TRAIN, total_batches=len(dataloader))
        self._metric_manager.reset()
        manual_metric_names: set[str] = set()
        logger.debug("Training phase started: epoch=%d", epoch)

        total_batches = len(dataloader)
        for batch_idx, batch_data in enumerate(dataloader):
            batch = move_to_device(batch_data, self._device)

            optimization_plan, before_step_context = self._configure_optimization_step(
                batch,
                batch_idx,
                total_batches,
                epoch,
            )

            if optimization_plan.zero_grad:
                # Optimizer is guaranteed to be non-None by train() method
                self._optimizer.zero_grad()  # type: ignore[union-attr]
                logger.debug(
                    "Gradients cleared: epoch=%d, batch=%d",
                    epoch,
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
                epoch,
                batch_idx,
                batch_metrics,
            )

            after_step_context: EventContext = {
                "battery": as_battery(self),
                "model": self._model,
                "optimizer": self._optimizer,
                "batch": batch,
                "batch_idx": batch_idx,
                "epoch": epoch,
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
            epoch,
            train_metrics,
        )
        return train_metrics
