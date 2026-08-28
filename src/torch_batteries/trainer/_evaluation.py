"""Validation and test workflows for ``torch_batteries.Battery``."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader

from torch_batteries.events import Event, EventContext
from torch_batteries.trainer.context import dataset_identity_context
from torch_batteries.trainer.types import TestResult, ValidationResult
from torch_batteries.utils.batch import get_batch_size
from torch_batteries.utils.device import move_to_device
from torch_batteries.utils.logging import get_logger
from torch_batteries.utils.progress import Phase, Progress, ProgressFactory

from ._state import BatteryStateMixin, as_battery

if TYPE_CHECKING:
    from torch_batteries.utils.progress.types import ProgressMetrics

logger = get_logger("trainer._evaluation")


class EvaluationMixin(BatteryStateMixin):
    """Implement validation and test workflows."""

    __slots__ = ()

    def _validate(
        self,
        val_loader: DataLoader | None = None,
        verbose: int = 1,
    ) -> ValidationResult:
        """Validate once with an explicit or DataPack-provided loader."""
        if val_loader is not None:
            return self._validate_with_loader(val_loader, verbose)
        with self._data_workflow("fit") as workflow:
            validation_loaders = workflow.loaders.loaders_for_phase("validation")
            validation_loader = next(iter(validation_loaders.values()), None)
            if validation_loader is None:
                msg = "The DataPack fit stage did not provide validation data."
                raise ValueError(msg)
            return self._validate_with_loader(validation_loader, verbose)

    def _validate_with_loader(
        self,
        val_loader: DataLoader,
        verbose: int = 1,
    ) -> ValidationResult:
        """Run one evaluation-only validation pass at epoch one."""
        self._validate_loader(val_loader, "Validation")
        logger.info("Validation started: batches=%d", len(val_loader))

        before_validation_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 1,
        }
        self._event_handler.call(Event.BEFORE_VALIDATION, before_validation_context)

        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(1)
        try:
            val_metrics = self._validate_epoch(val_loader, progress, 1)
        except BaseException:
            progress.abort()
            raise
        progress.end_epoch()
        progress.end_training()

        after_validation_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 1,
            "loss": val_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_metrics": val_metrics,
        }
        self._event_handler.call(Event.AFTER_VALIDATION, after_validation_context)

        result: ValidationResult = {"val_loss": val_metrics["loss"]}
        if len(val_metrics) > 1:
            result["val_metrics"] = {
                name: value for name, value in val_metrics.items() if name != "loss"
            }
        logger.info("Validation completed")
        return result

    def _test(
        self,
        test_loader: DataLoader | None = None,
        verbose: int = 1,
        *,
        dataset: str | None = None,
    ) -> TestResult | dict[str, TestResult]:
        """Evaluate once with an explicit or DataPack-provided test loader.

        Args:
            test_loader: Optional sized, non-empty test loader. When omitted, the
                attached DataPack must provide its test dataset.
            verbose: ``0`` for silent, ``1`` for a progress bar, or ``2`` for a summary.
            dataset: Optional name selecting one DataPack test dataset. It cannot be
                combined with an explicit loader.

        Returns:
            One test result for an explicit, selected, or bare dataset. A named
            dataset mapping returns results keyed by dataset name.
        """
        if test_loader is not None:
            if dataset is not None:
                msg = "dataset cannot be combined with an explicit test loader."
                raise ValueError(msg)
            return self._test_with_loader(test_loader, verbose)
        with self._data_workflow("test", dataset_name=dataset) as workflow:
            test_loaders = workflow.loaders.loaders_for_phase("test")
            results = {
                name: self._test_with_loader(
                    loader,
                    verbose,
                    dataset_name=name,
                )
                for name, loader in test_loaders.items()
            }
            if dataset is not None or not isinstance(workflow.loaders.test, Mapping):
                return next(iter(results.values()))
            return results

    def _test_with_loader(
        self,
        test_loader: DataLoader,
        verbose: int = 1,
        *,
        dataset_name: str | None = None,
    ) -> TestResult:
        """Evaluate the model once without gradient tracking.

        The model is placed in evaluation mode and ``Event.TEST_STEP`` runs for each
        batch. Callable metrics are sample-weighted; stateful and collected metrics
        compute once over the completed phase.

        Args:
            test_loader: Sized, non-empty test loader.
            verbose: ``0`` for silent, ``1`` for a progress bar, or ``2`` for a summary.

        Returns:
            Average test loss and, when present, named test metrics.

        Raises:
            ValueError: If the loader is empty or no test-step handler exists.
            TypeError: If a step returns an unsupported result structure.
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
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(Event.BEFORE_TEST, before_test_context)

        before_test_epoch_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 1,
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(Event.BEFORE_TEST_EPOCH, before_test_epoch_context)

        self._model.eval()

        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(1)
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
                        dataset_name=dataset_name,
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
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 1,
            "loss": test_loss,
            "test_loss": test_loss,
            "test_metrics": test_metrics_context,
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(Event.AFTER_TEST_EPOCH, after_test_epoch_context)

        after_test_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "loss": test_loss,
            "test_loss": test_loss,
            "test_metrics": test_metrics_context,
            **dataset_identity_context(dataset_name),
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
        *,
        dataset_name: str | None = None,
    ) -> None:
        """Process one test batch."""
        batch = move_to_device(batch_data, self._device)

        before_step_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "device": self._device,
            "phase": "test",
            "batch": batch,
            "batch_idx": batch_idx,
            "epoch": 1,
            **dataset_identity_context(dataset_name),
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
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "batch": batch,
            "batch_idx": batch_idx,
            "epoch": 1,
            "loss": loss.item(),
            "test_loss": loss.item(),
            "test_metrics": batch_metrics,
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(Event.AFTER_TEST_STEP, after_step_context)

        num_samples = get_batch_size(batch)
        progress.update(cast("ProgressMetrics", batch_metrics), num_samples)

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
            "battery": as_battery(self),
            "model": self._model,
            "epoch": epoch,
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
                    "battery": as_battery(self),
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "device": self._device,
                    "phase": "validation",
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": epoch,
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
                    epoch,
                    batch_idx,
                    batch_metrics,
                )

                after_step_context: EventContext = {
                    "battery": as_battery(self),
                    "model": self._model,
                    "batch": batch,
                    "batch_idx": batch_idx,
                    "epoch": epoch,
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
            "battery": as_battery(self),
            "model": self._model,
            "epoch": epoch,
            "loss": val_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_metrics": val_metrics,
        }
        self._event_handler.call(Event.AFTER_VALIDATION_EPOCH, after_val_epoch_context)

        return val_metrics
