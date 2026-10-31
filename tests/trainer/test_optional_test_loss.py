"""Test steps that report metrics without a loss."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, StepOutput, charge
from torch_batteries.callbacks import TerminateOnNonFinite

from .test_multi_dataset import NamedPack


class LosslessTestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.contexts: list[EventContext] = []

    @charge(Event.TRAIN_STEP)
    def train_step(self, context: EventContext) -> StepOutput:
        (values,) = context["batch"]
        return StepOutput(metrics={"mean": values.mean()})

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self.train_step(context)

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        (values,) = context["batch"]
        return StepOutput(metrics={"mean": values.mean()})

    @charge(Event.AFTER_TEST_STEP)
    @charge(Event.AFTER_TEST_EPOCH)
    @charge(Event.AFTER_TEST)
    def record(self, context: EventContext) -> None:
        self.contexts.append(context)


def loader(count: int = 4) -> DataLoader:
    return DataLoader(TensorDataset(torch.arange(count, dtype=torch.float32)[:, None]))


def test_test_without_loss_reports_metrics_only() -> None:
    """A lossless test omits test_loss from results and event contexts."""
    model = LosslessTestModel()
    battery = Battery(model, callbacks=[TerminateOnNonFinite()])

    result = battery.test(loader(), verbose=0)

    assert "test_loss" not in result
    assert result["test_metrics"] == {"mean": pytest.approx(1.5)}
    assert model.contexts
    assert all("test_loss" not in context for context in model.contexts)
    assert all("loss" not in context["test_metrics"] for context in model.contexts)


def test_named_test_without_loss_aggregates_metrics() -> None:
    """Named test datasets aggregate metrics without inventing a loss."""
    battery = Battery(LosslessTestModel(), data_pack=NamedPack())

    result = battery.test(verbose=0)

    assert "test_loss" not in result
    assert result["test_metrics"] == {
        "mean": pytest.approx((5 * 1.0 + 6 * 3.0) / 11),
        "small:mean": pytest.approx(1.0),
        "large:mean": pytest.approx(3.0),
    }


def test_training_requires_loss() -> None:
    """Training steps may not omit the loss."""
    battery = Battery(
        LosslessTestModel(), optimizer=torch.optim.SGD([nn.Parameter(torch.ones(1))])
    )

    with pytest.raises(
        ValueError, match="Training step must return StepOutput with a loss"
    ):
        battery.train(loader(), epochs=1, verbose=0)


def test_validation_requires_loss() -> None:
    """Validation steps may not omit the loss."""
    battery = Battery(LosslessTestModel())

    with pytest.raises(
        ValueError, match="Validation step must return StepOutput with a loss"
    ):
        battery.validate(loader(), verbose=0)


class MixedLossTestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        (values,) = context["batch"]
        if values.mean() == 3.0 or context["batch_idx"] > 0:
            return StepOutput(metrics={"mean": values.mean()})
        return StepOutput(loss=values.mean(), metrics={"mean": values.mean()})


def test_test_rejects_loss_on_only_some_batches() -> None:
    """Batches of one loader must agree on whether they return a loss."""
    battery = Battery(MixedLossTestModel())

    with pytest.raises(ValueError, match="all return a loss or all omit it"):
        battery.test(
            DataLoader(TensorDataset(torch.zeros(4, 1)), batch_size=2), verbose=0
        )


def test_named_test_rejects_loss_on_only_some_datasets() -> None:
    """Named test datasets must agree on whether they return a loss."""
    battery = Battery(MixedLossTestModel(), data_pack=NamedPack())

    with pytest.raises(ValueError, match="all return a loss or all omit it"):
        battery.test(verbose=0)


class LossMetricTestModel(LosslessTestModel):
    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        (values,) = context["batch"]
        return StepOutput(metrics={"loss": values.mean()})


def test_test_rejects_loss_as_metric_name() -> None:
    """A loss reported as a metric is not mistaken for the step loss."""
    battery = Battery(LossMetricTestModel())

    with pytest.raises(ValueError, match=r"return its loss as StepOutput\.loss"):
        battery.test(loader(), verbose=0)
