"""Tests for fit, standalone validation, and train-only behavior."""

import warnings
from pathlib import Path
from typing import cast

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
    FitResult,
    StepOutput,
    TrainResult,
    ValidationResult,
    charge,
)


def _loader() -> DataLoader:
    inputs = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    targets = inputs * 2
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


class WorkflowModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.validation_grad_states: list[bool] = []

    def _loss(self, context: EventContext) -> torch.Tensor:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return nn.functional.mse_loss(self.layer(inputs), targets)

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        return StepOutput(loss=self._loss(context))

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        self.validation_grad_states.append(torch.is_grad_enabled())
        return StepOutput(loss=self._loss(context), metrics={"score": 0.75})


class ValidationRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[Event, EventContext]] = []

    def _record(self, event: Event, context: EventContext) -> None:
        self.events.append((event, context.copy()))

    @charge(Event.BEFORE_VALIDATION)
    def before_validation(self, context: EventContext) -> None:
        self._record(Event.BEFORE_VALIDATION, context)

    @charge(Event.BEFORE_VALIDATION_EPOCH)
    def before_validation_epoch(self, context: EventContext) -> None:
        self._record(Event.BEFORE_VALIDATION_EPOCH, context)

    @charge(Event.AFTER_VALIDATION_EPOCH)
    def after_validation_epoch(self, context: EventContext) -> None:
        self._record(Event.AFTER_VALIDATION_EPOCH, context)

    @charge(Event.AFTER_VALIDATION)
    def after_validation(self, context: EventContext) -> None:
        self._record(Event.AFTER_VALIDATION, context)


class StopAfterEpoch:
    @charge(Event.AFTER_TRAIN_EPOCH)
    def stop(self, context: EventContext) -> None:
        if context["epoch"] == 2:
            context["battery"].request_stop("manual_limit")


class ValidationDataPack(DataPack):
    def __init__(self, *, include_validation: bool = True) -> None:
        self.dataset = _loader().dataset
        self.include_validation = include_validation
        self.stages: list[str] = []
        self.teardown_stages: list[str] = []

    @charge(Event.SETUP_DATA)
    def setup(self, context: DataContext) -> DatasetBundle:
        self.stages.append(context["stage"])
        return DatasetBundle(
            train=self.dataset,
            validation=self.dataset if self.include_validation else None,
        )

    @charge(Event.CONFIGURE_DATALOADER)
    def configure(self, _: DataContext) -> DataLoaderConfig:
        return DataLoaderConfig(batch_size=2)

    @charge(Event.TEARDOWN_DATA)
    def teardown(self, context: DataContext) -> None:
        self.teardown_stages.append(context["stage"])


def _battery(
    *,
    data_pack: DataPack | None = None,
    callbacks: list[object] | None = None,
    optimizer: bool = True,
) -> tuple[Battery, WorkflowModel]:
    model = WorkflowModel()
    configured_optimizer = (
        torch.optim.SGD(model.parameters(), lr=0.01) if optimizer else None
    )
    return (
        Battery(
            model,
            device="cpu",
            optimizer=configured_optimizer,
            callbacks=callbacks,
            data_pack=data_pack,
        ),
        model,
    )


def test_fit_runs_optional_validation_without_compatibility_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    battery, _ = _battery()

    with warnings.catch_warnings(record=True) as warnings_record:
        result: FitResult = battery.fit(_loader(), _loader(), verbose=0)

    assert warnings_record == []
    assert len(result["train_loss"]) == 1
    assert len(result["val_loss"]) == 1
    assert result["val_metrics"]["score"] == [0.75]
    assert "Validation through Battery.train()" not in caplog.text


def test_fit_without_validation_returns_empty_validation_histories() -> None:
    battery, _ = _battery()

    result = battery.fit(_loader(), verbose=0)

    assert len(result["train_loss"]) == 1
    assert result["val_loss"] == []
    assert result["val_metrics"] == {}


def test_train_only_runs_training() -> None:
    battery, _ = _battery()
    result: TrainResult = battery.train(_loader(), verbose=0)
    assert len(result["train_loss"]) == 1
    assert result["epochs_completed"] == 1
    assert result["optimizer_steps"] == 2
    assert result["stopped_early"] is False
    assert result["stop_reason"] is None


def test_validation_cadence_and_cumulative_metadata(tmp_path: Path) -> None:
    battery, model = _battery()
    first = battery.fit(
        _loader(), _loader(), epochs=3, verbose=0, validate_every_n_epochs=2
    )
    assert len(first["train_loss"]) == 3
    assert len(first["val_loss"]) == 1
    assert model.validation_grad_states == [False, False]
    assert (first["epochs_completed"], first["optimizer_steps"]) == (3, 6)

    checkpoint = tmp_path / "cadence.pth"
    battery.save_checkpoint(checkpoint)
    restored, _ = _battery()
    resumed = restored.fit(
        _loader(),
        _loader(),
        epochs=5,
        verbose=0,
        resume_from=checkpoint,
        validate_every_n_epochs=2,
    )
    assert len(resumed["train_loss"]) == 5
    assert len(resumed["val_loss"]) == 2
    assert (resumed["epochs_completed"], resumed["optimizer_steps"]) == (5, 10)


def test_stop_reason_is_reported() -> None:
    battery, _ = _battery(callbacks=[StopAfterEpoch()])
    result = battery.fit(_loader(), epochs=5, verbose=0)
    assert result["epochs_completed"] == 2
    assert result["optimizer_steps"] == 4
    assert result["stopped_early"] is True
    assert result["stop_reason"] == "manual_limit"


@pytest.mark.parametrize("cadence", [0, -1, 1.5, True])
def test_fit_rejects_invalid_validation_cadence(cadence: object) -> None:
    battery, _ = _battery()
    with pytest.raises(ValueError, match="validate_every_n_epochs"):
        battery.fit(_loader(), verbose=0, validate_every_n_epochs=cadence)  # type: ignore[arg-type]


def test_train_rejects_validation_loader() -> None:
    battery, _ = _battery()
    with pytest.raises(TypeError, match="val_loader"):
        battery.train(_loader(), val_loader=_loader(), verbose=0)  # type: ignore[call-arg]


def test_validate_runs_without_optimizer_or_gradients_and_dispatches_events() -> None:
    recorder = ValidationRecorder()
    battery, model = _battery(callbacks=[recorder], optimizer=False)

    result: ValidationResult = battery.validate(_loader(), verbose=0)

    assert isinstance(result["val_loss"], float)
    assert result["val_metrics"] == {"score": 0.75}
    assert model.validation_grad_states == [False, False]
    assert [event for event, _ in recorder.events] == [
        Event.BEFORE_VALIDATION,
        Event.BEFORE_VALIDATION_EPOCH,
        Event.AFTER_VALIDATION_EPOCH,
        Event.AFTER_VALIDATION,
    ]
    assert all(context["epoch"] == 1 for _, context in recorder.events)
    assert recorder.events[-2][1]["val_loss"] == result["val_loss"]
    assert recorder.events[-1][1]["val_loss"] == result["val_loss"]


def test_data_pack_supports_fit_and_required_validate_stages() -> None:
    data_pack = ValidationDataPack()
    battery, _ = _battery(data_pack=data_pack)

    fit_result = battery.fit(verbose=0)
    validation_result = battery.validate(verbose=0)

    assert len(fit_result["val_loss"]) == 1
    assert isinstance(validation_result["val_loss"], float)
    assert data_pack.stages == ["fit", "fit"]
    assert data_pack.teardown_stages == ["fit", "fit"]


def test_fit_allows_missing_data_pack_validation_but_validate_requires_it() -> None:
    data_pack = ValidationDataPack(include_validation=False)
    battery, _ = _battery(data_pack=data_pack)

    result = battery.fit(verbose=0)

    assert result["val_loss"] == []
    with pytest.raises(ValueError, match="fit stage did not provide validation"):
        battery.validate(verbose=0)
    assert data_pack.teardown_stages == ["fit", "fit"]


def test_fit_preserves_checkpoint_resume_behavior(tmp_path: Path) -> None:
    battery, _ = _battery()
    initial = battery.fit(_loader(), _loader(), verbose=0)
    checkpoint = tmp_path / "fit.pth"
    battery.save_checkpoint(checkpoint)
    restored, _ = _battery()

    resumed = restored.fit(
        _loader(),
        _loader(),
        epochs=2,
        verbose=0,
        resume_from=checkpoint,
    )

    assert len(initial["train_loss"]) == 1
    assert len(resumed["train_loss"]) == 2
    assert len(resumed["val_loss"]) == 2
