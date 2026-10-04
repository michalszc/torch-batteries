"""Tests for learning-rate scheduler callbacks."""

from typing import cast

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, StepOutput, charge
from torch_batteries.callbacks import GradientAccumulation, LearningRateScheduler


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return StepOutput(
            loss=cast("torch.Tensor", ((self.layer(inputs) - targets) ** 2).mean())
        )


def _loader(samples: int = 4) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(samples, 1), torch.zeros(samples, 1)),
        batch_size=1,
    )


def test_step_scheduler_advances_only_on_optimizer_steps() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    battery = Battery(
        model,
        optimizer=optimizer,
        callbacks=[
            GradientAccumulation(steps=2),
            LearningRateScheduler(scheduler, interval="step"),
        ],
    )

    battery.train(_loader(), verbose=0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)


def test_epoch_scheduler_advances_once_per_epoch() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    battery = Battery(
        model,
        optimizer=optimizer,
        callbacks=[LearningRateScheduler(scheduler)],
    )

    battery.train(_loader(1), epochs=2, verbose=0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)


def test_plateau_scheduler_uses_selected_validation_metric() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=0, factor=0.5)
    callback = LearningRateScheduler(
        scheduler,
        phase="validation",
        metric="loss",
    )

    callback.on_validation_end(EventContext(epoch=0, val_metrics={"loss": 1.0}))
    callback.on_validation_end(EventContext(epoch=1, val_metrics={"loss": 1.0}))

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_rejects_removed_stage_alias() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(TypeError, match="unexpected keyword argument 'stage'"):
        LearningRateScheduler(
            ReduceLROnPlateau(optimizer),
            stage="validation",
            metric="loss",
        )  # type: ignore[call-arg]


def test_rejects_phase_and_stage() -> None:
    """The removed stage keyword is unavailable alongside phase."""
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(TypeError, match="unexpected keyword argument 'stage'"):
        LearningRateScheduler(
            ReduceLROnPlateau(optimizer),
            phase="train",
            stage="validation",
            metric="loss",
        )  # type: ignore[call-arg]


def test_validates_plateau_and_ordinary_scheduler_configuration() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="requires interval='epoch'"):
        LearningRateScheduler(ReduceLROnPlateau(optimizer), interval="step")
    with pytest.raises(ValueError, match="only supported"):
        LearningRateScheduler(StepLR(optimizer, 1), phase="validation", metric="loss")


def test_scheduler_state_round_trip() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = LearningRateScheduler(StepLR(optimizer, 1))
    optimizer.step()
    callback.on_train_epoch_end(EventContext(epoch=0, train_metrics={"loss": 1.0}))
    state = callback.state_dict()

    restored_model = _Model()
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored = LearningRateScheduler(StepLR(restored_optimizer, 1))
    restored.load_state_dict(state)

    assert "phase" in state
    assert "stage" not in state
    assert restored.scheduler.last_epoch == callback.scheduler.last_epoch


def test_rejects_removed_scheduler_stage_state() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = LearningRateScheduler(
        ReduceLROnPlateau(optimizer),
        phase="validation",
        metric="loss",
    )
    state = callback.state_dict()
    state["stage"] = state.pop("phase")

    with pytest.raises(ValueError, match="requires 'phase'"):
        callback.load_state_dict(state)


def test_rejects_removed_scheduler_val_state() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = LearningRateScheduler(
        ReduceLROnPlateau(optimizer),
        phase="validation",
        metric="loss",
    )
    state = callback.state_dict()
    state["phase"] = "val"

    with pytest.raises(ValueError, match="phase must be"):
        callback.load_state_dict(state)


@pytest.mark.parametrize(
    "phase_keys",
    [
        {},
    ],
)
def test_rejects_ambiguous_scheduler_phase_state(
    phase_keys: dict[str, object],
) -> None:
    """Checkpoint state identifies its monitoring phase with exactly one key."""
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = LearningRateScheduler(StepLR(optimizer, 1))
    state = callback.state_dict()
    state.pop("phase")
    state.update(phase_keys)

    with pytest.raises(ValueError, match="requires 'phase'"):
        callback.load_state_dict(state)


def test_rejects_mismatched_scheduler_phase_state() -> None:
    """The restored monitoring phase must match callback configuration."""
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = LearningRateScheduler(
        ReduceLROnPlateau(optimizer),
        phase="train",
        metric="loss",
    )
    state = callback.state_dict()
    state["phase"] = "validation"

    with pytest.raises(ValueError, match="does not match"):
        callback.load_state_dict(state)


def test_rejects_invalid_interval_and_missing_plateau_metric() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="interval must be"):
        LearningRateScheduler(
            StepLR(optimizer, 1),
            interval="batch",  # type: ignore[arg-type]
        )

    callback = LearningRateScheduler(
        ReduceLROnPlateau(optimizer),
        phase="train",
        metric="loss",
    )
    with pytest.raises(ValueError, match="is unavailable"):
        callback.on_train_epoch_end(EventContext(epoch=0, train_metrics={}))


def test_validation_plateau_requires_validation_and_metric() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = LearningRateScheduler(
        ReduceLROnPlateau(optimizer),
        phase="validation",
        metric="loss",
    )

    with pytest.raises(ValueError, match="validation loader"):
        callback.on_train_end(EventContext(epoch=0))
    with pytest.raises(ValueError, match="is unavailable"):
        callback.on_validation_end(EventContext(epoch=0, val_metrics={}))


def test_rejects_invalid_scheduler_checkpoint_state() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = LearningRateScheduler(StepLR(optimizer, 1))

    with pytest.raises(ValueError, match="requires 'phase'"):
        callback.load_state_dict(
            {
                "interval": "step",
                "stage": None,
                "metric": None,
                "scheduler": {},
                "stepped_epochs": [],
            }
        )
    with pytest.raises(TypeError, match="Invalid LearningRateScheduler"):
        callback.load_state_dict(
            {
                "interval": "epoch",
                "phase": None,
                "metric": None,
                "scheduler": None,
                "stepped_epochs": None,
            }
        )
    with pytest.raises(TypeError, match="Invalid LearningRateScheduler"):
        callback.load_state_dict(
            {
                "interval": "epoch",
                "phase": None,
                "metric": None,
                "scheduler": {},
                "stepped_epochs": ["one"],
            }
        )


def test_scheduler_ignores_events_for_other_routes() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ordinary = LearningRateScheduler(StepLR(optimizer, 1))
    validation_plateau = LearningRateScheduler(
        ReduceLROnPlateau(optimizer),
        phase="validation",
        metric="loss",
    )

    ordinary.on_validation_end(EventContext(epoch=0, val_metrics={"loss": 1.0}))
    validation_plateau.on_train_epoch_end(
        EventContext(epoch=0, train_metrics={"loss": 1.0})
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
