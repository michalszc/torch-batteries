"""Phase-specific and structured metric behavior."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import (
    Battery,
    CollectedMetric,
    Event,
    EventContext,
    MetricSpec,
    StepOutput,
    charge,
)


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (predictions.argmax(dim=1) == targets).float().mean()


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (predictions - targets).abs().mean()


class StructuredModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def _step(self, context: EventContext) -> StepOutput:
        values = context["batch"][0]
        return StepOutput(
            loss=self.weight * 0 + 1,
            predictions={
                "classification": torch.stack((values, -values), dim=1),
                "regression": values + 1,
            },
            targets={
                "classification": torch.zeros(len(values), dtype=torch.long),
                "regression": values,
            },
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


def _loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.ones(4)), batch_size=2)


def test_phase_specific_structured_metric_specs() -> None:
    model = StructuredModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        metrics={
            "train": {
                "accuracy": MetricSpec(
                    accuracy, predictions="classification", targets="classification"
                )
            },
            "validation": {
                "mae": MetricSpec(
                    CollectedMetric(mae),
                    predictions="regression",
                    targets="regression",
                )
            },
            "test": {
                "accuracy": MetricSpec(
                    accuracy, predictions="classification", targets="classification"
                ),
                "mae": MetricSpec(mae, predictions="regression", targets="regression"),
            },
        },
    )

    fit = battery.fit(_loader(), _loader(), verbose=0)
    test = battery.test(_loader(), verbose=0)

    assert fit["train_metrics"] == {"accuracy": [1.0]}
    assert fit["val_metrics"] == {"mae": [1.0]}
    assert test["test_metrics"] == {"accuracy": 1.0, "mae": 1.0}


def test_metric_spec_rejects_incomplete_or_blank_selectors() -> None:
    with pytest.raises(ValueError, match="both predictions and targets"):
        MetricSpec(mae, predictions="regression")
    with pytest.raises(ValueError, match="non-blank"):
        MetricSpec(mae, predictions=" ", targets="regression")
    with pytest.raises(TypeError, match="must be strings"):
        MetricSpec(mae, predictions=1, targets="regression")  # type: ignore[arg-type]


def test_structured_metric_requires_selected_key() -> None:
    model = StructuredModel()
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        metrics={"mae": mae},
    )
    with pytest.raises(ValueError, match="requires a predictions selector"):
        battery.train(_loader(), verbose=0)


def test_metric_spec_rejects_missing_selected_key() -> None:
    model = StructuredModel()
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        metrics={"mae": MetricSpec(mae, predictions="missing", targets="regression")},
    )
    with pytest.raises(ValueError, match="missing predictions key 'missing'"):
        battery.train(_loader(), verbose=0)
