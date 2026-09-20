"""Validation and test workflows for ``torch_batteries.Battery``."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader

from torch_batteries.data import BatchScheduleConfig
from torch_batteries.events import Event, EventContext
from torch_batteries.trainer.context import dataset_identity_context
from torch_batteries.trainer.types import TestResult, ValidationResult
from torch_batteries.utils.batch import get_batch_size
from torch_batteries.utils.device import move_to_device
from torch_batteries.utils.logging import get_logger
from torch_batteries.utils.metrics import PhaseMetricManager
from torch_batteries.utils.metrics._dataset_totals import DatasetMetricTotals
from torch_batteries.utils.progress import (
    Phase,
    Progress,
    ProgressFactory,
    SilentProgress,
)

from ._batch_schedule import scheduled_batches
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
            if not validation_loaders:
                msg = "The DataPack fit stage did not provide validation data."
                raise ValueError(msg)
            return self._validate_with_loaders(
                validation_loaders,
                verbose,
                schedule=workflow.datasets.batch_schedule,
                named_datasets=isinstance(workflow.loaders.validation, Mapping),
            )

    def _validate_with_loader(
        self,
        val_loader: DataLoader,
        verbose: int = 1,
    ) -> ValidationResult:
        """Run one evaluation-only validation pass at epoch one."""
        return self._validate_with_loaders({"default": val_loader}, verbose)

    def _validate_with_loaders(
        self,
        loaders: dict[str, DataLoader],
        verbose: int = 1,
        *,
        schedule: BatchScheduleConfig | None = None,
        named_datasets: bool = False,
    ) -> ValidationResult:
        """Run one evaluation-only validation pass over all loaders."""
        schedule = schedule or BatchScheduleConfig()
        for name, loader in loaders.items():
            self._validate_loader(loader, f"Validation '{name}'")
            self._restore_loader_generator_state(
                "validation", loader, dataset_name=name if named_datasets else None
            )
        logger.info("Validation started: batches=%d", sum(map(len, loaders.values())))

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
            val_metrics = self._validate_epoch(
                loaders, progress, 1, schedule, named_datasets=named_datasets
            )
        except BaseException:
            progress.abort()
            raise
        for name, loader in loaders.items():
            self._capture_loader_generator_state(
                "validation", loader, dataset_name=name if named_datasets else None
            )
        progress.end_epoch()
        progress.end_training()

        after_validation_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 1,
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
    ) -> TestResult:
        """Evaluate once with an explicit or DataPack-provided test loader.

        Args:
            test_loader: Optional sized, non-empty test loader. When omitted, the
                attached DataPack must provide its test dataset.
            verbose: ``0`` for silent, ``1`` for a progress bar, or ``2`` for a summary.
            dataset: Optional name selecting one DataPack test dataset. It cannot be
                combined with an explicit loader.

        Returns:
            One aggregate test result. Named metrics use ``dataset:metric`` keys.
        """
        if test_loader is not None:
            if dataset is not None:
                msg = "dataset cannot be combined with an explicit test loader."
                raise ValueError(msg)
            return self._test_with_loader(test_loader, verbose)[0]
        with self._data_workflow("test", dataset_name=dataset) as workflow:
            test_loaders = workflow.loaders.loaders_for_phase("test")
            shared_progress: Progress | None = None
            if len(test_loaders) > 1:
                shared_progress = ProgressFactory.create(
                    verbose=verbose, total_epochs=1
                )
                shared_progress.start_epoch(1)
                shared_progress.start_phase(
                    Phase.TEST, total_batches=sum(map(len, test_loaders.values()))
                )
            results: dict[str, tuple[TestResult, int]] = {}
            try:
                for index, (name, loader) in enumerate(test_loaders.items()):
                    results[name] = self._test_with_loader(
                        loader,
                        verbose,
                        dataset_name=name,
                        reset_metrics=index == 0,
                        named_metrics=len(test_loaders) > 1,
                        shared_progress=shared_progress,
                    )
            except BaseException:
                if shared_progress is not None:
                    shared_progress.abort()
                raise
            if shared_progress is not None:
                shared_progress.end_phase()
                shared_progress.end_epoch()
            if len(results) == 1:
                return next(iter(results.values()))[0]
            total_samples = sum(samples for _, samples in results.values())
            aggregate: dict[str, float] = {}
            for name, (result, samples) in results.items():
                values = {"loss": result["test_loss"], **result.get("test_metrics", {})}
                for metric, value in values.items():
                    aggregate[metric] = aggregate.get(metric, 0.0) + value * samples
                    aggregate[f"{name}:{metric}"] = value
            metrics = {
                name: value / total_samples
                for name, value in aggregate.items()
                if ":" not in name
            }
            metrics.update(
                {name: value for name, value in aggregate.items() if ":" in name}
            )
            metrics.update(self._manager_for_phase("test").compute())
            aggregate_result: TestResult = {
                "test_loss": metrics.pop("loss"),
                "test_metrics": metrics,
            }
            aggregate_context: EventContext = {
                "battery": as_battery(self),
                "model": self._model,
                "optimizer": self._optimizer,
                "epoch": 1,
                "test_loss": aggregate_result["test_loss"],
                "test_metrics": {"loss": aggregate_result["test_loss"], **metrics},
            }
            self._event_handler.call(Event.AFTER_TEST_EPOCH, aggregate_context)
            self._event_handler.call(Event.AFTER_TEST, aggregate_context)
            return aggregate_result

    def _test_with_loader(  # noqa: PLR0913
        self,
        test_loader: DataLoader,
        verbose: int = 1,
        *,
        dataset_name: str | None = None,
        reset_metrics: bool = True,
        named_metrics: bool = False,
        shared_progress: Progress | None = None,
    ) -> tuple[TestResult, int]:
        """Evaluate the model once without gradient tracking.

        The model is placed in evaluation mode and ``Event.TEST_STEP`` runs for each
        batch. Callable metrics are sample-weighted; stateful and collected metrics
        compute once over the completed phase.

        Args:
            test_loader: Sized, non-empty test loader.
            verbose: ``0`` for silent, ``1`` for a progress bar, or ``2`` for a summary.
            dataset_name: Name included in dataset-specific event contexts.
            reset_metrics: Reset aggregate metric state before this loader.
            named_metrics: Maintain isolated metric state for this dataset.
            shared_progress: Optional progress tracker shared by named loaders.

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
        self._restore_loader_generator_state(
            "test", test_loader, dataset_name=dataset_name
        )
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

        progress = shared_progress or ProgressFactory.create(
            verbose=verbose, total_epochs=1
        )
        metric_progress = progress if shared_progress is None else SilentProgress()
        if shared_progress is None:
            progress.start_epoch(1)
            progress.start_phase(Phase.TEST, total_batches=len(test_loader))
        else:
            metric_progress.start_phase(Phase.TEST, total_batches=len(test_loader))
        metric_manager = self._manager_for_phase("test")
        if reset_metrics:
            metric_manager.reset()
        dataset_manager = (
            self._manager_for_dataset("test", dataset_name)
            if named_metrics and dataset_name is not None
            else None
        )
        if dataset_manager is not None:
            dataset_manager.reset()
        manual_metric_names: set[str] = set()
        total_samples = 0
        logger.debug("Test phase started: epoch=1")

        try:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    total_samples += self._test_batch(
                        batch_data,
                        batch_idx,
                        progress,
                        manual_metric_names,
                        dataset_name=dataset_name,
                        dataset_manager=dataset_manager,
                        named_metrics=named_metrics,
                        metric_progress=metric_progress,
                    )
        except BaseException:
            progress.abort()
            raise

        self._capture_loader_generator_state(
            "test", test_loader, dataset_name=dataset_name
        )

        test_metrics = metric_progress.end_phase()
        if shared_progress is None:
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
                for name, value in (
                    dataset_manager.compute()
                    if dataset_manager is not None
                    else metric_manager.compute()
                ).items()
                if name not in manual_metric_names
            }
        )

        after_test_epoch_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 1,
            "test_loss": test_loss,
            "test_metrics": test_metrics_context,
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(Event.AFTER_TEST_EPOCH, after_test_epoch_context)

        after_test_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
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
        return results, total_samples

    def _test_batch(  # noqa: PLR0913
        self,
        batch_data: Any,
        batch_idx: int,
        progress: Progress,
        manual_metric_names: set[str],
        *,
        dataset_name: str | None = None,
        dataset_manager: PhaseMetricManager | None = None,
        named_metrics: bool = False,
        metric_progress: Progress | None = None,
    ) -> int:
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
            self._manager_for_phase("test").update(predictions, targets)
            if predictions is not None and targets is not None
            else {}
        )
        if (
            predictions is not None
            and targets is not None
            and dataset_manager is not None
        ):
            dataset_manager.update(predictions, targets)
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
            "test_loss": loss.item(),
            "test_metrics": (
                {
                    **batch_metrics,
                    **{
                        f"{dataset_name}:{name}": value
                        for name, value in batch_metrics.items()
                    },
                }
                if named_metrics
                else batch_metrics
            ),
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(Event.AFTER_TEST_STEP, after_step_context)

        num_samples = get_batch_size(batch)
        progress.update(
            cast("ProgressMetrics", batch_metrics),
            num_samples,
            dataset_name=dataset_name if named_metrics else None,
        )
        if metric_progress is not None and metric_progress is not progress:
            metric_progress.update(cast("ProgressMetrics", batch_metrics), num_samples)
        return num_samples

    def _validate_epoch(
        self,
        loaders: DataLoader | dict[str, DataLoader],
        progress: Progress,
        epoch: int,
        schedule: BatchScheduleConfig | None = None,
        *,
        named_datasets: bool = False,
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
        if isinstance(loaders, DataLoader):
            loaders = {"default": loaders}
        schedule = schedule or BatchScheduleConfig()

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

        progress.start_phase(
            Phase.VALIDATION, total_batches=sum(map(len, loaders.values()))
        )
        metric_manager = self._manager_for_phase("validation")
        metric_manager.reset()
        dataset_managers = (
            {name: self._manager_for_dataset("validation", name) for name in loaders}
            if len(loaders) > 1
            else {}
        )
        for manager in dataset_managers.values():
            manager.reset()
        manual_metric_names: set[str] = set()
        dataset_totals = DatasetMetricTotals()

        with torch.no_grad():
            for batch_idx, (dataset_name, batch_data) in enumerate(
                scheduled_batches(loaders, schedule, epoch)
            ):
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
                    **dataset_identity_context(
                        dataset_name if named_datasets else None
                    ),
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
                    metric_manager.update(predictions, targets)
                    if predictions is not None and targets is not None
                    else {}
                )
                if predictions is not None and targets is not None and dataset_managers:
                    dataset_managers[dataset_name].update(predictions, targets)
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
                    "val_loss": loss.item(),
                    "val_metrics": (
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
                    **dataset_identity_context(
                        dataset_name if named_datasets else None
                    ),
                }
                self._event_handler.call(
                    Event.AFTER_VALIDATION_STEP, after_step_context
                )

                num_samples = get_batch_size(batch)
                progress.update(
                    cast("ProgressMetrics", batch_metrics),
                    num_samples,
                    dataset_name=dataset_name if len(loaders) > 1 else None,
                )
                dataset_totals.update(dataset_name, batch_metrics, num_samples)

        avg_metrics = progress.end_phase()
        val_metrics = (
            avg_metrics if isinstance(avg_metrics, dict) else {"loss": avg_metrics}
        )
        val_metrics.update(
            {
                name: value
                for name, value in metric_manager.compute().items()
                if name not in manual_metric_names
            }
        )
        if len(loaders) > 1:
            val_metrics.update(dataset_totals.compute())
            for dataset_name, manager in dataset_managers.items():
                val_metrics.update(
                    {
                        f"{dataset_name}:{name}": value
                        for name, value in manager.compute().items()
                        if name not in manual_metric_names
                    }
                )

        # Trigger AFTER_VALIDATION_EPOCH event
        after_val_epoch_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
            "val_metrics": val_metrics,
        }
        self._event_handler.call(Event.AFTER_VALIDATION_EPOCH, after_val_epoch_context)

        return val_metrics
