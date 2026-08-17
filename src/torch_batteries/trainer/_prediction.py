"""Prediction workflows for ``torch_batteries.Battery``."""

from collections.abc import Generator, Iterator, Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader

from torch_batteries.events import Event, EventContext
from torch_batteries.trainer.context import dataset_identity_context
from torch_batteries.trainer.types import PredictResult
from torch_batteries.utils.device import move_to_device
from torch_batteries.utils.logging import get_logger
from torch_batteries.utils.prediction import concatenate_predictions
from torch_batteries.utils.progress import Phase, Progress, ProgressFactory

from ._state import BatteryStateMixin, as_battery

logger = get_logger("trainer._prediction")


class PredictionMixin(BatteryStateMixin):
    """Implement eager and streaming prediction workflows."""

    __slots__ = ()

    def _predict(
        self,
        data_loader: DataLoader | None = None,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
        concatenate: bool = False,
        dataset: str | None = None,
    ) -> PredictResult | dict[str, PredictResult]:
        """Collect predictions using an explicit or DataPack-provided loader.

        Args:
            data_loader: Optional sized, non-empty prediction loader. When omitted,
                the attached DataPack must provide its prediction dataset.
            verbose: ``0`` for silent, ``1`` for a progress bar, or ``2`` for a summary.
            move_to_cpu: Recursively detach tensor outputs and move them to CPU.
            concatenate: Recursively concatenate matching outputs along their first
                dimension while retaining nested container structure.
            dataset: Optional name selecting one DataPack prediction dataset. It
                cannot be combined with an explicit loader.

        Returns:
            One prediction result for an explicit, selected, or bare dataset. A named
            dataset mapping returns results keyed by dataset name.
        """
        if data_loader is not None:
            if dataset is not None:
                msg = "dataset cannot be combined with an explicit prediction loader."
                raise ValueError(msg)
            return self._predict_with_loader(
                data_loader,
                verbose,
                move_to_cpu=move_to_cpu,
                concatenate=concatenate,
            )
        with self._data_workflow("predict", dataset_name=dataset) as workflow:
            prediction_loaders = workflow.loaders.loaders_for_phase("predict")
            results = {
                name: self._predict_with_loader(
                    loader,
                    verbose,
                    move_to_cpu=move_to_cpu,
                    concatenate=concatenate,
                    dataset_name=name,
                )
                for name, loader in prediction_loaders.items()
            }
            if dataset is not None or not isinstance(workflow.loaders.predict, Mapping):
                return next(iter(results.values()))
            return results

    def _predict_with_loader(
        self,
        data_loader: DataLoader,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
        concatenate: bool = False,
        dataset_name: str | None = None,
    ) -> PredictResult:
        """Collect predictions from one evaluation-mode pass over a loader.

        The default result contains one user-defined output per batch. Structured
        outputs can be recursively moved to CPU and concatenated without changing
        matching dictionary, tuple, named-tuple, or list containers.

        Args:
            data_loader: Sized, non-empty prediction loader.
            verbose: ``0`` for silent, ``1`` for a progress bar, or ``2`` for a summary.
            move_to_cpu: Recursively detach tensor outputs and move them to CPU.
                This is useful when predictions should not retain accelerator memory.
            concatenate: Recursively concatenate matching tensor outputs along their
                first dimension. Nested dictionaries, tuples, named tuples, and lists
                retain their structure.

        Returns:
            Mapping containing either a list of batch outputs or one recursively
            concatenated structured output.

        Raises:
            ValueError: If the loader is empty, no handler exists, or output shapes
                cannot be concatenated.
            TypeError: If concatenated batch structures do not match.
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
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(Event.BEFORE_PREDICT, before_predict_context)

        before_predict_epoch_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 1,
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(
            Event.BEFORE_PREDICT_EPOCH, before_predict_epoch_context
        )

        self._model.eval()
        predictions: list[Any] = []

        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(1)
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
                        dataset_name=dataset_name,
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
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "epoch": 1,
            "predictions": prediction_output,
            "prediction_batches": len(predictions),
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(Event.AFTER_PREDICT_EPOCH, after_predict_epoch_context)

        after_predict_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "predictions": prediction_output,
            "prediction_batches": len(predictions),
            **dataset_identity_context(dataset_name),
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
        data_loader: DataLoader | None = None,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
        dataset: str | None = None,
    ) -> Generator[Any]:
        """Stream predictions using an explicit or DataPack-provided loader.

        Args:
            data_loader: Optional sized, non-empty prediction loader. When omitted,
                the attached DataPack must provide its prediction dataset.
            verbose: ``0`` for silent, ``1`` for a progress bar, or ``2`` for a summary.
            move_to_cpu: Recursively detach tensor outputs and move them to CPU before
                yielding them.
            dataset: Optional name selecting one DataPack prediction dataset. It is
                required when the DataPack provides multiple prediction datasets.

        Yields:
            One user-defined prediction-step output at a time.

        Note:
            Iteration must finish for final prediction and DataPack teardown events to
            run.
        """
        if data_loader is not None:
            if dataset is not None:
                msg = "dataset cannot be combined with an explicit prediction loader."
                raise ValueError(msg)
            yield from self._predict_iter_with_loader(
                data_loader,
                verbose,
                move_to_cpu=move_to_cpu,
            )
            return
        with self._data_workflow(
            "predict",
            dataset_name=dataset,
        ) as workflow:
            prediction_loaders = workflow.loaders.loaders_for_phase("predict")
            if dataset is None and len(prediction_loaders) > 1:
                available = ", ".join(repr(name) for name in prediction_loaders)
                msg = (
                    "predict_iter() requires dataset= when multiple prediction "
                    f"datasets are configured. Available datasets: {available}."
                )
                raise ValueError(msg)
            name, loader = next(iter(prediction_loaders.items()))
            yield from self._predict_iter_with_loader(
                loader,
                verbose,
                move_to_cpu=move_to_cpu,
                dataset_name=name,
            )

    def _predict_iter_with_loader(
        self,
        data_loader: DataLoader,
        verbose: int = 1,
        *,
        move_to_cpu: bool = False,
        dataset_name: str | None = None,
    ) -> Iterator[Any]:
        """Yield prediction batches without retaining the complete result.

        Lifecycle events still receive the number of yielded batches, but not an
        accumulated ``predictions`` value. Iteration must finish for final prediction
        events to run.

        Args:
            data_loader: Sized, non-empty prediction loader.
            verbose: ``0`` for silent, ``1`` for a progress bar, or ``2`` for a summary.
            move_to_cpu: Recursively detach tensor outputs and move them to CPU
                before yielding them.

        Yields:
            One user-defined prediction-step output at a time.

        Raises:
            ValueError: If the loader is empty or no predict-step handler exists.
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
            dataset_name=dataset_name,
        )

    def _prediction_iterator(
        self,
        data_loader: DataLoader,
        verbose: int,
        *,
        move_to_cpu: bool,
        dataset_name: str | None = None,
    ) -> Iterator[Any]:
        """Run lazy prediction lifecycle and yield each non-None output."""
        before_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(Event.BEFORE_PREDICT, before_context)
        self._event_handler.call(
            Event.BEFORE_PREDICT_EPOCH,
            {**before_context, "epoch": 1},
        )
        self._model.eval()
        progress = ProgressFactory.create(verbose=verbose, total_epochs=1)
        progress.start_epoch(1)
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
                        dataset_name=dataset_name,
                    )
                    processed_batches += 1
                    if prediction is not None:
                        yield prediction
            progress.end_phase()
            progress.end_epoch()
            completion_context: EventContext = {
                "battery": as_battery(self),
                "model": self._model,
                "optimizer": self._optimizer,
                "epoch": 1,
                "prediction_batches": processed_batches,
                **dataset_identity_context(dataset_name),
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
        dataset_name: str | None = None,
    ) -> Any:
        """Process one prediction batch."""
        batch = move_to_device(batch_data, self._device)

        before_step_context: EventContext = {
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "device": self._device,
            "phase": "predict",
            "batch": batch,
            "batch_idx": batch_idx,
            "epoch": 1,
            **dataset_identity_context(dataset_name),
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
            "battery": as_battery(self),
            "model": self._model,
            "optimizer": self._optimizer,
            "batch": batch,
            "batch_idx": batch_idx,
            "epoch": 1,
            "predictions": prediction,
            **dataset_identity_context(dataset_name),
        }
        self._event_handler.call(Event.AFTER_PREDICT_STEP, after_step_context)

        progress.update()
        return prediction
