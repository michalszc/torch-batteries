"""Tests for non-finite loss and metric termination."""

from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, StepOutput, charge
from torch_batteries.callbacks import TerminateOnNonFinite


class _Model(nn.Module):
    def __init__(
        self,
        *,
        loss: float = 1.0,
        metric: float | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.loss = loss
        self.metric = metric

    def _step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        predictions = inputs * self.weight
        loss = self.weight * 0 + self.loss
        metrics: dict[str, float | torch.Tensor] = (
            {} if self.metric is None else {"score": self.metric}
        )
        return StepOutput(
            loss=loss,
            predictions=predictions,
            targets=targets,
            metrics=metrics,
        )

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        return self._step(context)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self._step(context)

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        return self._step(context)


class _InfiniteStatefulMetric:
    def reset(self) -> None:
        pass

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        del predictions, targets

    def compute(self) -> float:
        return float("inf")


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)),
        batch_size=1,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"check_loss": 1},
        {"check_metrics": 0},
        {"check_loss": False, "check_metrics": False},
    ],
)
def test_configuration_is_strict(kwargs: dict[str, Any]) -> None:
    exception = ValueError if kwargs.get("check_loss") is False else TypeError

    with pytest.raises(exception):
        TerminateOnNonFinite(**kwargs)


def test_non_finite_training_loss_stops_before_backward_and_optimizer() -> None:
    model = _Model(loss=float("nan"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    battery = Battery(
        model,
        optimizer=optimizer,
        callbacks=[TerminateOnNonFinite()],
    )

    with (
        patch.object(optimizer, "step", wraps=optimizer.step) as optimizer_step,
        pytest.raises(
            FloatingPointError,
            match=r"name='loss'.*phase='train'.*epoch=1.*batch=0",
        ),
    ):
        battery.train(_loader(), verbose=0)

    optimizer_step.assert_not_called()
    assert model.weight.grad is None


@pytest.mark.parametrize("workflow", ["validate", "test"])
def test_non_finite_evaluation_loss_is_rejected(workflow: str) -> None:
    model = _Model(loss=float("inf"))
    battery = Battery(model, callbacks=[TerminateOnNonFinite()])
    phase = "validation" if workflow == "validate" else "test"

    with pytest.raises(
        FloatingPointError,
        match=rf"name='loss'.*phase='{phase}'.*batch=0",
    ):
        getattr(battery, workflow)(_loader(), verbose=0)


def test_non_finite_named_batch_metric_is_rejected() -> None:
    model = _Model(metric=float("nan"))
    battery = Battery(model, callbacks=[TerminateOnNonFinite()])

    with pytest.raises(
        FloatingPointError,
        match=r"name='score'.*phase='validation'.*batch=0",
    ):
        battery.validate(_loader(), verbose=0)


def test_non_finite_final_stateful_metric_is_rejected() -> None:
    model = _Model()
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        metrics={"stateful": _InfiniteStatefulMetric()},
        callbacks=[TerminateOnNonFinite()],
    )

    with pytest.raises(
        FloatingPointError,
        match=r"name='stateful'.*phase='train'.*epoch=1",
    ):
        battery.train(_loader(), verbose=0)


def test_loss_name_always_follows_check_loss() -> None:
    context = EventContext(
        val_loss=1.0,
        val_metrics={"loss": float("inf"), "score": 1.0},
        epoch=1,
        batch_idx=0,
    )

    TerminateOnNonFinite(check_loss=False, check_metrics=True).on_step_end(context)
    with pytest.raises(FloatingPointError, match="name='loss'"):
        TerminateOnNonFinite(check_loss=True, check_metrics=False).on_step_end(context)


def test_metric_check_can_be_disabled_independently() -> None:
    context = EventContext(
        val_loss=1.0,
        val_metrics={"loss": 1.0, "score": float("inf")},
        epoch=1,
        batch_idx=0,
    )

    TerminateOnNonFinite(check_metrics=False).on_step_end(context)


def test_checkpoint_configuration_must_match() -> None:
    source = TerminateOnNonFinite(check_metrics=False)
    state = source.state_dict()
    restored = TerminateOnNonFinite(check_metrics=False)

    restored.load_state_dict(state)
    with pytest.raises(ValueError, match="does not match"):
        TerminateOnNonFinite(check_loss=False).load_state_dict(state)
