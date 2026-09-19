"""Training workflows for ``torch_batteries.Battery``."""

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader

from torch_batteries.data import BatchScheduleConfig
from torch_batteries.events import Event, EventContext, OptimizationStep
from torch_batteries.trainer.context import (
    copy_history_context,
    dataset_identity_context,
)
from torch_batteries.trainer.types import FitResult, TrainResult
from torch_batteries.utils.batch import get_batch_size
from torch_batteries.utils.device import move_to_device
from torch_batteries.utils.logging import get_logger
from torch_batteries.utils.metrics._dataset_totals import DatasetMetricTotals
from torch_batteries.utils.progress import Phase, Progress, ProgressFactory

from ._batch_schedule import scheduled_batches
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
        validate_every_n_epochs: int = 1,
    ) -> FitResult:
        """Fit the model with optional per-epoch validation.

        Args:
            train_loader: Optional sized, non-empty training loader.
            val_loader: Optional validation loader for direct-loader mode.
            epochs: Positive epoch count or resume target.
            verbose: ``0`` for silent, ``1`` for bars, or ``2`` for summaries.
            resume_from: Optional full checkpoint restored before data setup.
            resume_epochs_mode: ``"total"`` or ``"additional"``.
            validate_every_n_epochs: Run validation on absolute epoch multiples.

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
            run_validation=True,
            validate_every_n_epochs=validate_every_n_epochs,
        )

    def train(
        self,
        train_loader: DataLoader | None = None,
        epochs: int = 1,
        verbose: int = 1,
        *,
        resume_from: str | Path | None = None,
        resume_epochs_mode: str = "total",
    ) -> TrainResult:
        """Train the model for one or more epochs.

        Passing ``train_loader`` selects direct-loader mode. When it is omitted, the
        attached DataPack supplies training data. A checkpoint
        passed through ``resume_from`` is restored before DataPack setup.

        Args:
            train_loader: Optional sized, non-empty training loader.
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
        result = self._run_training_workflow(
            train_loader,
            None,
            epochs,
            verbose,
            resume_from=resume_from,
            resume_epochs_mode=resume_epochs_mode,
            run_validation=False,
            validate_every_n_epochs=1,
        )
        return {
            "train_loss": result["train_loss"],
            "train_metrics": result["train_metrics"],
            "epochs_completed": result["epochs_completed"],
            "optimizer_steps": result["optimizer_steps"],
            "stopped_early": result["stopped_early"],
            "stop_reason": result["stop_reason"],
        }

    def _run_training_workflow(  # noqa: PLR0913
        self,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        epochs: int,
        verbose: int,
        *,
        resume_from: str | Path | None,
        resume_epochs_mode: str,
        run_validation: bool,
        validate_every_n_epochs: int,
    ) -> FitResult:
        """Resolve loaders and run the shared training engine."""
        if epochs <= 0:
            msg = "epochs must be greater than zero."
            raise ValueError(msg)
        if (
            isinstance(validate_every_n_epochs, bool)
            or not isinstance(validate_every_n_epochs, int)
            or validate_every_n_epochs < 1
        ):
            msg = "validate_every_n_epochs must be a positive integer."
            raise ValueError(msg)
        if resume_epochs_mode not in {"total", "additional"}:
            logger.error("Unsupported resume epochs mode: %s", resume_epochs_mode)
            msg = "resume_epochs_mode must be 'total' or 'additional'."
            raise ValueError(msg)
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        if train_loader is not None:
            return self._train_with_loaders(
                {"default": train_loader},
                {"default": val_loader} if val_loader is not None else {},
                epochs,
                verbose,
                resume_epochs_mode=resume_epochs_mode,
                validate_every_n_epochs=validate_every_n_epochs,
            )
        if val_loader is not None:
            msg = (
                "An explicit validation loader cannot be combined with an implicit "
                "DataPack training loader."
            )
            raise ValueError(msg)
        with self._data_workflow("fit") as workflow:
            train_loaders = workflow.loaders.loaders_for_phase("train")
            validation_loaders = (
                workflow.loaders.loaders_for_phase("validation")
                if run_validation
                else {}
            )
            return self._train_with_loaders(
                train_loaders,
                validation_loaders,
                epochs,
                verbose,
                resume_epochs_mode=resume_epochs_mode,
                validate_every_n_epochs=validate_every_n_epochs,
                train_schedule=workflow.datasets.train_batch_schedule,
                validation_schedule=workflow.datasets.validation_batch_schedule,
                named_train=isinstance(workflow.loaders.train, Mapping),
                named_validation=isinstance(workflow.loaders.validation, Mapping),
            )

    def _train_with_loaders(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        train_loaders: dict[str, DataLoader],
        validation_loaders: dict[str, DataLoader] | None = None,
        epochs: int = 1,
        verbose: int = 1,
        *,
        resume_epochs_mode: str = "total",
        validate_every_n_epochs: int = 1,
        train_schedule: BatchScheduleConfig | None = None,
        validation_schedule: BatchScheduleConfig | None = None,
        named_train: bool = False,
        named_validation: bool = False,
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

        Returns:
            Per-epoch loss histories and named metric histories. Validation entries
            remain empty when no validation loader is supplied.

        Raises:
            ValueError: If inputs, handlers, resume mode, or checkpoint state are
                incompatible.
            TypeError: If a step result has an unsupported structure.
        """
        validation_loaders = validation_loaders or {}
        train_schedule = train_schedule or BatchScheduleConfig()
        validation_schedule = validation_schedule or BatchScheduleConfig()
        train_loader = next(iter(train_loaders.values()))
        val_loader = next(iter(validation_loaders.values()), None)
        self._validate_train_inputs(train_loader, val_loader)
        for name, loader in train_loaders.items():
            self._validate_loader(loader, f"Training '{name}'")
            self._restore_loader_generator_state(
                "train", loader, dataset_name=name if named_train else None
            )
        for name, loader in validation_loaders.items():
            self._validate_loader(loader, f"Validation '{name}'")
            self._restore_loader_generator_state(
                "validation", loader, dataset_name=name if named_validation else None
            )
        resumed = self._resume_loaded
        self._stop_training = False
        self._stop_reason = None
        if not resumed:
            self._optimizer_step_idx = 0
            self._last_completed_epoch = 0
            self._train_results = {
                "train_loss": [],
                "val_loss": [],
                "train_metrics": {},
                "val_metrics": {},
                "epochs_completed": 0,
                "optimizer_steps": 0,
                "stopped_early": False,
                "stop_reason": None,
            }
        logger.info(
            "Training started: epochs=%d, train_batches=%d, validation=%s",
            epochs,
            sum(map(len, train_loaders.values())),
            bool(validation_loaders),
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
                train_metrics = self._train_epoch(
                    train_loaders,
                    progress,
                    epoch,
                    train_schedule,
                    named_datasets=named_train,
                )
            except BaseException:
                progress.abort()
                raise
            for name, loader in train_loaders.items():
                self._capture_loader_generator_state(
                    "train", loader, dataset_name=name if named_train else None
                )
            results["train_loss"].append(train_metrics["loss"])

            for key, value in train_metrics.items():
                if key != "loss":
                    if key not in results["train_metrics"]:
                        results["train_metrics"][key] = []
                    results["train_metrics"][key].append(value)
            self._last_completed_epoch = epoch
            results["epochs_completed"] = epoch
            results["optimizer_steps"] = self._optimizer_step_idx
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

            if validation_loaders and epoch % validate_every_n_epochs == 0:
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
                    val_metrics = self._validate_epoch(
                        validation_loaders,
                        progress,
                        epoch,
                        validation_schedule,
                        named_datasets=named_validation,
                    )
                except BaseException:
                    progress.abort()
                    raise
                for name, loader in validation_loaders.items():
                    self._capture_loader_generator_state(
                        "validation",
                        loader,
                        dataset_name=name if named_validation else None,
                    )
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

        results["epochs_completed"] = self._last_completed_epoch
        results["optimizer_steps"] = self._optimizer_step_idx
        results["stopped_early"] = self._stop_training
        results["stop_reason"] = self._stop_reason

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
        self,
        loaders: dict[str, DataLoader],
        progress: Progress,
        epoch: int,
        schedule: BatchScheduleConfig,
        *,
        named_datasets: bool,
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

        total_batches = sum(map(len, loaders.values()))
        progress.start_phase(Phase.TRAIN, total_batches=total_batches)
        metric_manager = self._manager_for_phase("train")
        metric_manager.reset()
        dataset_managers = (
            {name: self._manager_for_dataset("train", name) for name in loaders}
            if len(loaders) > 1
            else {}
        )
        for manager in dataset_managers.values():
            manager.reset()
        manual_metric_names: set[str] = set()
        dataset_totals = DatasetMetricTotals()
        logger.debug("Training phase started: epoch=%d", epoch)

        for batch_idx, (dataset_name, batch_data) in enumerate(
            scheduled_batches(loaders, schedule, epoch)
        ):
            batch = move_to_device(batch_data, self._device)

            optimization_plan, before_step_context = self._configure_optimization_step(
                batch,
                batch_idx,
                total_batches,
                epoch,
            )
            before_step_context.update(
                dataset_identity_context(dataset_name if named_datasets else None)
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
                metric_manager.update(predictions, targets)
                if predictions is not None and targets is not None
                else {}
            )
            if predictions is not None and targets is not None and dataset_managers:
                dataset_managers[dataset_name].update(predictions, targets)
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
                "train_loss": loss.item(),
                "train_metrics": (
                    {
                        **batch_metrics,
                        **{
                            f"{dataset_name}:{name}": value
                            for name, value in batch_metrics.items()
                        },
                    }
                    if len(loaders) > 1
                    else batch_metrics
                ),
                "optimizer_step": optimizer_step,
                "optimizer_step_idx": self._optimizer_step_idx,
                "optimization_plan": optimization_plan,
                **dataset_identity_context(dataset_name if named_datasets else None),
            }
            self._event_handler.call(Event.AFTER_TRAIN_STEP, after_step_context)

            num_samples = get_batch_size(batch)
            progress.update(cast("ProgressMetrics", batch_metrics), num_samples)
            dataset_totals.update(dataset_name, batch_metrics, num_samples)

        avg_metrics = progress.end_phase()
        train_metrics = (
            avg_metrics if isinstance(avg_metrics, dict) else {"loss": avg_metrics}
        )
        train_metrics.update(
            {
                name: value
                for name, value in metric_manager.compute().items()
                if name not in manual_metric_names
            }
        )
        if len(loaders) > 1:
            train_metrics.update(dataset_totals.compute())
            for dataset_name, manager in dataset_managers.items():
                train_metrics.update(
                    {
                        f"{dataset_name}:{name}": value
                        for name, value in manager.compute().items()
                        if name not in manual_metric_names
                    }
                )
        logger.debug(
            "Training phase completed: epoch=%d, metrics=%s",
            epoch,
            train_metrics,
        )
        return train_metrics
