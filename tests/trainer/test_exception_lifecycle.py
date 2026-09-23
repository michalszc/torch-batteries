"""Tests for public workflow exception lifecycle handling."""

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import (
    Battery,
    DataContext,
    DataLoaderConfig,
    DataPack,
    DatasetBundle,
    Event,
    EventContext,
    StepOutput,
    charge,
)
from torch_batteries.callbacks import ExperimentTrackingCallback
from torch_batteries.tracking import ExperimentTracker


class _WorkflowModel(nn.Module):
    def __init__(self, failure: RuntimeError | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.failure = failure

    def _result(self, context: EventContext) -> torch.Tensor:
        if self.failure is not None:
            raise self.failure
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return ((inputs * self.weight - targets) ** 2).mean()

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        return StepOutput(loss=self._result(context))

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return StepOutput(loss=self._result(context))

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        return StepOutput(loss=self._result(context))

    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> torch.Tensor:
        if self.failure is not None:
            raise self.failure
        inputs, _ = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return inputs * self.weight


class _ExceptionRecorder:
    def __init__(self) -> None:
        self.contexts: list[EventContext] = []

    @charge(Event.ON_EXCEPTION)
    def record(self, context: EventContext) -> None:
        self.contexts.append(context.copy())


class _FailingCallback:
    def __init__(self, failure: RuntimeError) -> None:
        self.failure = failure

    @charge(Event.BEFORE_TRAIN_EPOCH)
    def fail(self, _: EventContext) -> None:
        raise self.failure


class _FailingExceptionHandler:
    @charge(Event.ON_EXCEPTION)
    def fail_cleanup(self, _: EventContext) -> None:
        msg = "cleanup failed"
        raise RuntimeError(msg)


class _FailingTeardownDataPack(DataPack):
    @charge(Event.SETUP_DATA)
    def setup(self, _: DataContext) -> DatasetBundle:
        return DatasetBundle(train=_loader().dataset)

    @charge(Event.CONFIGURE_DATALOADER)
    def configure_loader(self, _: DataContext) -> DataLoaderConfig:
        return DataLoaderConfig(batch_size=1)

    @charge(Event.TEARDOWN_DATA)
    def teardown(self, _: DataContext) -> None:
        msg = "teardown failed"
        raise RuntimeError(msg)


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)),
        batch_size=1,
    )


def _execute_workflow(battery: Battery, workflow: str) -> None:
    result = getattr(battery, workflow)(_loader(), verbose=0)
    if workflow == "predict_iter":
        next(result)


@pytest.mark.parametrize(
    "workflow",
    ["train", "fit", "validate", "test", "predict", "predict_iter"],
)
def test_each_public_workflow_dispatches_original_exception_once(workflow: str) -> None:
    failure = RuntimeError(f"{workflow} failed")
    recorder = _ExceptionRecorder()
    model = _WorkflowModel(failure)
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[recorder],
    )

    with pytest.raises(RuntimeError, match=f"{workflow} failed") as caught:
        _execute_workflow(battery, workflow)

    assert caught.value is failure
    assert len(recorder.contexts) == 1
    assert recorder.contexts[0] == {
        "battery": battery,
        "model": model,
        "optimizer": battery.optimizer,
        "exception": failure,
    }


def test_callback_failure_dispatches_exception() -> None:
    failure = RuntimeError("callback failed")
    recorder = _ExceptionRecorder()
    model = _WorkflowModel()
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[_FailingCallback(failure), recorder],
    )

    with pytest.raises(RuntimeError, match="callback failed") as caught:
        battery.train(_loader(), verbose=0)

    assert caught.value is failure
    assert recorder.contexts[0]["exception"] is failure


def test_exception_handler_failures_are_logged_without_stopping_later_cleanup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = RuntimeError("model failed")
    recorder = _ExceptionRecorder()
    model = _WorkflowModel(failure)
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[_FailingExceptionHandler(), recorder],
    )

    with pytest.raises(RuntimeError, match="model failed") as caught:
        battery.train(_loader(), verbose=0)

    assert caught.value is failure
    assert recorder.contexts[0]["exception"] is failure
    assert "ON_EXCEPTION handler" in caplog.text
    assert "cleanup failed" in caplog.text


def test_data_pack_teardown_failure_dispatches_exception() -> None:
    recorder = _ExceptionRecorder()
    model = _WorkflowModel()
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[recorder],
        data_pack=_FailingTeardownDataPack(),
    )

    with pytest.raises(RuntimeError, match="teardown failed") as caught:
        battery.train(verbose=0)

    assert recorder.contexts[0]["exception"] is caught.value


def test_tracker_failure_closes_initialized_run_with_original_error() -> None:
    failure = RuntimeError("tracker failed")
    tracker = MagicMock(spec=ExperimentTracker)
    tracker.is_initialized = True
    tracker.log_metrics.side_effect = failure
    tracking = ExperimentTrackingCallback(tracker=tracker)
    recorder = _ExceptionRecorder()
    model = _WorkflowModel()
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[tracking, recorder],
    )

    with pytest.raises(RuntimeError, match="tracker failed") as caught:
        battery.train(_loader(), verbose=0)

    assert caught.value is failure
    tracker.finish.assert_called_once_with(exit_code=1)
    tracker.log_model.assert_not_called()
    assert recorder.contexts[0]["exception"] is failure


def test_streaming_exhaustion_and_close_do_not_dispatch_exception() -> None:
    recorder = _ExceptionRecorder()
    model = _WorkflowModel()
    battery = Battery(model, callbacks=[recorder])

    assert len(list(battery.predict_iter(_loader(), verbose=0))) == 1
    iterator = battery.predict_iter(_loader(), verbose=0)
    next(iterator)
    iterator.close()

    assert recorder.contexts == []
