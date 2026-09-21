"""Measure CPU workflow overhead separately from loader and charged-step time.

The residual includes event dispatch, metrics, progress, and the required
autograd and optimizer work during training.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from time import perf_counter_ns
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import (
    Battery,
    DataContext,
    DataPack,
    DatasetBundle,
    Event,
    EventContext,
    StepOutput,
    charge,
)

PHASES = ("train", "validation", "test", "predict", "fit")
MODES = ("direct", "named")


class TimedLoader(DataLoader):
    """Account for batch retrieval without changing the actual DataLoader path."""

    def __init__(self, dataset: TensorDataset, batch_size: int) -> None:
        super().__init__(dataset, batch_size=batch_size, num_workers=0)
        self.elapsed_ns = 0
        self.batches = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        start = perf_counter_ns()
        iterator = super().__iter__()
        self.elapsed_ns += perf_counter_ns() - start
        while True:
            start = perf_counter_ns()
            try:
                batch = next(iterator)
            except StopIteration:
                self.elapsed_ns += perf_counter_ns() - start
                return
            self.elapsed_ns += perf_counter_ns() - start
            self.batches += 1
            yield batch


class TinyModel(nn.Module):
    """Minimize work in charged steps while retaining real train semantics."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.step_ns = 0
        self.steps = 0

    def _loss_step(self, _: EventContext) -> StepOutput:
        start = perf_counter_ns()
        result = StepOutput(loss=self.weight * 0 + 1)
        self.step_ns += perf_counter_ns() - start
        self.steps += 1
        return result

    @charge(Event.TRAIN_STEP)
    def train_step(self, context: EventContext) -> StepOutput:
        return self._loss_step(context)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self._loss_step(context)

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        return self._loss_step(context)

    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> torch.Tensor:
        start = perf_counter_ns()
        result = cast("torch.Tensor", context["batch"][0])
        self.step_ns += perf_counter_ns() - start
        self.steps += 1
        return result


class NamedPack(DataPack):
    """Exercise DataPack resolution and named loader dispatch in each workflow."""

    def __init__(self, datasets: dict[str, TensorDataset], batch_size: int) -> None:
        self.datasets = datasets
        self.batch_size = batch_size
        self.loaders: list[TimedLoader] = []

    @charge(Event.SETUP_DATA)
    def setup(self, _: DataContext) -> DatasetBundle:
        return DatasetBundle(
            train=self.datasets,
            validation=self.datasets,
            test=self.datasets,
            predict=self.datasets,
        )

    @charge(Event.CONFIGURE_DATALOADER)
    def configure(self, context: DataContext) -> TimedLoader:
        loader = TimedLoader(self.datasets[context["dataset_name"]], self.batch_size)
        self.loaders.append(loader)
        return loader


@dataclass(frozen=True)
class Measurement:
    """Median elapsed and isolated costs, normalized by completed batches."""

    total_ms: float
    loader_us_per_batch: float
    step_us_per_batch: float
    overhead_us_per_batch: float
    batches: int


def _one_run(
    phase: str, mode: str, batches: int, batch_size: int
) -> tuple[int, int, int, int]:
    dataset = TensorDataset(torch.zeros(batches * batch_size, 1))
    pack: NamedPack | None = None
    loaders: list[TimedLoader]
    if mode == "named":
        pack = NamedPack({"first": dataset, "second": dataset}, batch_size)
        loaders = pack.loaders
    else:
        loaders = [TimedLoader(dataset, batch_size)]
    model = TinyModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        data_pack=pack,
    )
    assert battery.device.type == "cpu"
    loader = None if pack is not None else loaders[0]
    start = perf_counter_ns()
    if phase == "train":
        battery.train(loader, epochs=1, verbose=0)
    elif phase == "validation":
        battery.validate(loader, verbose=0)
    elif phase == "test":
        battery.test(loader, verbose=0)
    elif phase == "predict":
        battery.predict(loader, verbose=0)
    elif phase == "fit":
        battery.fit(loader, loader, epochs=1, verbose=0)
    else:
        msg = f"Unknown phase: {phase}"
        raise ValueError(msg)
    elapsed = perf_counter_ns() - start
    loader_ns = sum(item.elapsed_ns for item in loaders)
    completed = sum(item.batches for item in loaders)
    assert completed == model.steps
    assert completed == batches * (2 if mode == "named" else 1) * (
        2 if phase == "fit" else 1
    )
    assert elapsed >= loader_ns + model.step_ns
    return elapsed, loader_ns, model.step_ns, completed


def measure(  # noqa: PLR0913
    phase: str,
    mode: str,
    *,
    batches: int,
    batch_size: int,
    repeats: int,
    warmups: int,
) -> Measurement:
    """Measure one workload with warmups and median per-batch cost."""
    if phase not in PHASES or mode not in MODES:
        msg = "Unknown phase or mode"
        raise ValueError(msg)
    if min(batches, batch_size, repeats) < 1 or warmups < 0:
        msg = "Batch count, batch size, and repeats must be positive"
        raise ValueError(msg)
    for _ in range(warmups):
        _one_run(phase, mode, batches, batch_size)
    samples = [_one_run(phase, mode, batches, batch_size) for _ in range(repeats)]
    completed = samples[0][3]
    return Measurement(
        total_ms=statistics.median(sample[0] for sample in samples) / 1_000_000,
        loader_us_per_batch=statistics.median(
            sample[1] / sample[3] for sample in samples
        )
        / 1_000,
        step_us_per_batch=statistics.median(sample[2] / sample[3] for sample in samples)
        / 1_000,
        overhead_us_per_batch=statistics.median(
            (sample[0] - sample[1] - sample[2]) / sample[3] for sample in samples
        )
        / 1_000,
        batches=completed,
    )
