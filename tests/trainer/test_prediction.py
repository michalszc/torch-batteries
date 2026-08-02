"""Tests for structured and streaming prediction."""

from typing import Protocol, cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, charge


class _PredictionModel(nn.Module):
    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> dict[str, object]:
        inputs, _ = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return {
            "scores": inputs * 2,
            "pair": (inputs, [inputs + 1]),
        }


class _CompletionRecorder:
    def __init__(self) -> None:
        self.completed: list[int] = []

    @charge(Event.AFTER_PREDICT)
    def after_predict(self, context: EventContext) -> None:
        self.completed.append(context["prediction_batches"])


class _Closable(Protocol):
    def close(self) -> None: ...


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.arange(4.0).reshape(4, 1), torch.zeros(4, 1)),
        batch_size=2,
    )


def test_predict_defaults_to_legacy_batch_list() -> None:
    result = Battery(_PredictionModel(), device="cpu").predict(_loader(), verbose=0)

    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 2


def test_predict_concatenates_nested_outputs_on_cpu() -> None:
    result = Battery(_PredictionModel(), device="cpu").predict(
        _loader(),
        verbose=0,
        move_to_cpu=True,
        concatenate=True,
    )
    predictions = cast("dict[str, object]", result["predictions"])
    scores = cast("torch.Tensor", predictions["scores"])
    pair = cast("tuple[torch.Tensor, list[torch.Tensor]]", predictions["pair"])

    assert scores.shape == (4, 1)
    assert scores.device.type == "cpu"
    assert pair[0].shape == (4, 1)
    assert pair[1][0].shape == (4, 1)


def test_predict_iter_yields_batches_and_completes_lifecycle() -> None:
    recorder = _CompletionRecorder()
    battery = Battery(_PredictionModel(), device="cpu", callbacks=[recorder])

    outputs = list(battery.predict_iter(_loader(), verbose=0, move_to_cpu=True))

    assert len(outputs) == 2
    assert recorder.completed == [2]


def test_predict_iter_early_close_does_not_fire_completion() -> None:
    recorder = _CompletionRecorder()
    battery = Battery(_PredictionModel(), device="cpu", callbacks=[recorder])
    iterator = battery.predict_iter(_loader(), verbose=0)

    next(iterator)
    cast("_Closable", iterator).close()

    assert recorder.completed == []


def test_predict_iter_requires_prediction_handler() -> None:
    battery = Battery(nn.Linear(1, 1), device="cpu")

    with pytest.raises(ValueError, match="PREDICT_STEP"):
        list(battery.predict_iter(_loader(), verbose=0))
