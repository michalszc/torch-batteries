"""Integration tests for documented event context contracts."""

from typing import Any, cast

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, StepOutput, charge

COMMON_FIELDS = {"battery", "model"}

EXPECTED_FIELDS: dict[Event, set[str]] = {
    Event.BEFORE_TRAIN: COMMON_FIELDS | {"optimizer"},
    Event.AFTER_TRAIN: COMMON_FIELDS
    | {
        "optimizer",
        "epoch",
        "train_metrics",
        "val_metrics",
        "history_train_loss",
        "history_val_loss",
        "history_train_metrics",
        "history_val_metrics",
    },
    Event.BEFORE_TRAIN_EPOCH: COMMON_FIELDS | {"optimizer", "epoch"},
    Event.AFTER_TRAIN_EPOCH: COMMON_FIELDS
    | {
        "optimizer",
        "epoch",
        "train_metrics",
        "history_train_loss",
        "history_val_loss",
        "history_train_metrics",
        "history_val_metrics",
    },
    Event.BEFORE_TRAIN_STEP: COMMON_FIELDS
    | {"optimizer", "batch", "batch_idx", "epoch"},
    Event.TRAIN_STEP: COMMON_FIELDS | {"optimizer", "batch", "batch_idx", "epoch"},
    Event.AFTER_TRAIN_STEP: COMMON_FIELDS
    | {
        "optimizer",
        "batch",
        "batch_idx",
        "epoch",
        "loss",
        "train_loss",
        "train_metrics",
    },
    Event.BEFORE_VALIDATION: COMMON_FIELDS
    | {
        "optimizer",
        "epoch",
        "train_metrics",
        "history_train_loss",
        "history_val_loss",
        "history_train_metrics",
        "history_val_metrics",
    },
    Event.AFTER_VALIDATION: COMMON_FIELDS
    | {
        "optimizer",
        "epoch",
        "train_metrics",
        "val_metrics",
        "history_train_loss",
        "history_val_loss",
        "history_train_metrics",
        "history_val_metrics",
    },
    Event.BEFORE_VALIDATION_EPOCH: COMMON_FIELDS | {"epoch"},
    Event.AFTER_VALIDATION_EPOCH: COMMON_FIELDS | {"epoch", "val_metrics"},
    Event.BEFORE_VALIDATION_STEP: COMMON_FIELDS | {"batch", "batch_idx", "epoch"},
    Event.VALIDATION_STEP: COMMON_FIELDS | {"batch", "batch_idx", "epoch"},
    Event.AFTER_VALIDATION_STEP: COMMON_FIELDS
    | {
        "batch",
        "batch_idx",
        "epoch",
        "loss",
        "val_loss",
        "val_metrics",
    },
    Event.BEFORE_TEST: COMMON_FIELDS | {"optimizer"},
    Event.AFTER_TEST: COMMON_FIELDS
    | {"optimizer", "loss", "test_loss", "test_metrics"},
    Event.BEFORE_TEST_EPOCH: COMMON_FIELDS | {"optimizer", "epoch"},
    Event.AFTER_TEST_EPOCH: COMMON_FIELDS
    | {"optimizer", "epoch", "loss", "test_loss", "test_metrics"},
    Event.BEFORE_TEST_STEP: COMMON_FIELDS
    | {"optimizer", "batch", "batch_idx", "epoch"},
    Event.TEST_STEP: COMMON_FIELDS | {"optimizer", "batch", "batch_idx", "epoch"},
    Event.AFTER_TEST_STEP: COMMON_FIELDS
    | {
        "optimizer",
        "batch",
        "batch_idx",
        "epoch",
        "loss",
        "test_loss",
        "test_metrics",
    },
    Event.BEFORE_PREDICT: COMMON_FIELDS | {"optimizer"},
    Event.AFTER_PREDICT: COMMON_FIELDS | {"optimizer", "predictions"},
    Event.BEFORE_PREDICT_EPOCH: COMMON_FIELDS | {"optimizer", "epoch"},
    Event.AFTER_PREDICT_EPOCH: COMMON_FIELDS | {"optimizer", "epoch", "predictions"},
    Event.BEFORE_PREDICT_STEP: COMMON_FIELDS
    | {"optimizer", "batch", "batch_idx", "epoch"},
    Event.PREDICT_STEP: COMMON_FIELDS | {"optimizer", "batch", "batch_idx", "epoch"},
    Event.AFTER_PREDICT_STEP: COMMON_FIELDS
    | {"optimizer", "batch", "batch_idx", "epoch", "predictions"},
}

MODEL_STEP_EVENTS = {
    Event.TRAIN_STEP,
    Event.VALIDATION_STEP,
    Event.TEST_STEP,
    Event.PREDICT_STEP,
}


class EventRecorder:
    """Dynamically register a recorder for every callback-compatible event."""

    def __init__(self, records: dict[Event, list[EventContext]]) -> None:
        for event in Event:
            if event in MODEL_STEP_EVENTS:
                continue

            def record(context: EventContext, event: Event = event) -> None:
                records[event].append(context.copy())

            handler = charge(event)(record)
            setattr(self, f"record_{event.value}", handler)


class ContractModel(nn.Module):
    """Small model that records contexts for model-specific step events."""

    def __init__(self, records: dict[Event, list[EventContext]]) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.records = records

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)  # type: ignore[no-any-return]

    def _supervised_step(self, context: EventContext, event: Event) -> StepOutput:
        self.records[event].append(context.copy())
        inputs, targets = context["batch"]
        predictions = self(inputs)
        return StepOutput(
            loss=nn.functional.mse_loss(predictions, targets),
            predictions=predictions,
            targets=targets,
            metrics={"manual": 2.0},
        )

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        return self._supervised_step(context, Event.TRAIN_STEP)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self._supervised_step(context, Event.VALIDATION_STEP)

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        return self._supervised_step(context, Event.TEST_STEP)

    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> torch.Tensor:
        self.records[Event.PREDICT_STEP].append(context.copy())
        inputs, _ = context["batch"]
        return self(inputs)  # type: ignore[no-any-return]


def _make_loader() -> DataLoader:
    inputs = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 2.0], [2.0, 3.0]])
    targets = torch.tensor([[1.0], [1.5], [2.0], [2.5]])
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


def _assert_batch_on_device(batch: Any, device: torch.device) -> None:
    assert isinstance(batch, list | tuple)
    assert batch
    assert all(isinstance(value, torch.Tensor) for value in batch)
    assert all(value.device == device for value in batch)


def test_all_documented_event_context_contracts() -> None:  # noqa: PLR0915
    """Every event receives its documented fields and representative values."""
    records: dict[Event, list[EventContext]] = {event: [] for event in Event}
    recorder = EventRecorder(records)
    model = ContractModel(records)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    battery = Battery(
        model,
        device="cpu",
        optimizer=optimizer,
        metrics={"mae": lambda pred, target: (pred - target).abs().mean()},
        callbacks=[recorder],
    )
    loader = _make_loader()

    battery.train(loader, loader, epochs=1, verbose=0)
    battery.test(loader, verbose=0)
    prediction_result = battery.predict(loader, verbose=0)

    assert len(Event) == 28
    assert set(EXPECTED_FIELDS) == set(Event)
    assert all(records[event] for event in Event)

    for event, event_records in records.items():
        for context in event_records:
            assert EXPECTED_FIELDS[event] <= context.keys()
            assert context["battery"] is battery
            assert context["model"] is model
            if "optimizer" in EXPECTED_FIELDS[event]:
                assert context["optimizer"] is optimizer
            if "epoch" in EXPECTED_FIELDS[event]:
                assert context["epoch"] == 0
            if "batch_idx" in EXPECTED_FIELDS[event]:
                assert context["batch_idx"] in {0, 1}
            if "batch" in EXPECTED_FIELDS[event]:
                _assert_batch_on_device(context["batch"], battery.device)

    phase_contracts = (
        (Event.AFTER_TRAIN_STEP, "train_loss", "train_metrics"),
        (Event.AFTER_VALIDATION_STEP, "val_loss", "val_metrics"),
        (Event.AFTER_TEST_STEP, "test_loss", "test_metrics"),
    )
    for event, loss_key, metrics_key in phase_contracts:
        for context in records[event]:
            dynamic_context = cast("dict[str, Any]", context)
            assert isinstance(dynamic_context["loss"], float)
            assert dynamic_context[loss_key] == dynamic_context["loss"]
            metrics = dynamic_context[metrics_key]
            assert metrics["loss"] == context["loss"]
            assert isinstance(metrics["mae"], float)
            assert metrics["manual"] == 2.0

    before_validation = records[Event.BEFORE_VALIDATION][0]
    assert len(before_validation["history_train_loss"]) == 1
    assert before_validation["history_val_loss"] == []
    assert before_validation["history_train_metrics"] == {
        "mae": [before_validation["train_metrics"]["mae"]],
        "manual": [2.0],
    }

    after_validation = records[Event.AFTER_VALIDATION][0]
    assert len(after_validation["history_train_loss"]) == 1
    assert len(after_validation["history_val_loss"]) == 1
    assert after_validation["history_val_metrics"]["manual"] == [2.0]

    after_train = records[Event.AFTER_TRAIN][0]
    assert after_train["train_metrics"]["manual"] == 2.0
    assert after_train["val_metrics"]["manual"] == 2.0
    assert len(after_train["history_train_loss"]) == 1
    assert len(after_train["history_val_loss"]) == 1

    for event in (Event.AFTER_TEST_EPOCH, Event.AFTER_TEST):
        context = records[event][0]
        assert context["test_loss"] == context["loss"]
        assert context["test_metrics"]["loss"] == context["loss"]
        assert context["test_metrics"]["manual"] == 2.0

    step_predictions = records[Event.AFTER_PREDICT_STEP]
    for context in step_predictions:
        prediction = context["predictions"]
        assert isinstance(prediction, torch.Tensor)
        assert prediction.shape == (2, 1)
    accumulated = prediction_result["predictions"]
    assert len(accumulated) == 2
    assert records[Event.AFTER_PREDICT_EPOCH][0]["predictions"] is accumulated
    assert records[Event.AFTER_PREDICT][0]["predictions"] is accumulated

    records[Event.AFTER_TRAIN].clear()
    battery.train(loader, epochs=1, verbose=0)
    assert "val_metrics" not in records[Event.AFTER_TRAIN][0]
