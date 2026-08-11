"""Tests for charged DataPack dispatch and DataLoader construction."""

from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from torch_batteries import (
    DataContext,
    DataLoaderConfig,
    DataPack,
    DatasetBundle,
    Event,
    EventContext,
    charge,
)
from torch_batteries.data.handler import DataPackHandler
from torch_batteries.data.loader import materialize_dataloader
from torch_batteries.events import EventHandler


class ExampleDataPack(DataPack):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.dataset = TensorDataset(torch.arange(6))

    @charge(Event.PREPARE_DATA)
    def prepare(self, _: DataContext) -> None:
        self.calls.append("prepare")

    @charge(Event.SETUP_DATA)
    def setup(self, _: DataContext) -> DatasetBundle:
        self.calls.append("setup")
        return DatasetBundle(train=self.dataset)

    @charge(Event.CONFIGURE_DATALOADER)
    def configure(self, context: DataContext) -> DataLoaderConfig:
        self.calls.append(f"loader:{context['phase']}")
        return DataLoaderConfig(batch_size=2)

    @charge(Event.TEARDOWN_DATA)
    def teardown(self, _: DataContext) -> None:
        self.calls.append("teardown")


def _context(data_pack: DataPack, phase: str = "train") -> DataContext:
    return {
        "data_pack": data_pack,
        "stage": "fit",
        "phase": phase,  # type: ignore[typeddict-item]
        "device": torch.device("cpu"),
        "seed": 7,
        "generator": torch.Generator().manual_seed(7),
    }


def test_handler_dispatches_data_lifecycle() -> None:
    data_pack = ExampleDataPack()
    handler = DataPackHandler(data_pack)
    context = _context(data_pack)

    handler.call(Event.PREPARE_DATA, context)
    datasets = handler.setup(context)
    assert datasets.train is data_pack.dataset
    context["datasets"] = datasets
    context["dataset"] = data_pack.dataset
    loader = handler.build_loader(context, data_pack.dataset)
    handler.call(Event.TEARDOWN_DATA, context)

    assert len(loader) == 3
    assert data_pack.calls == ["prepare", "setup", "loader:train", "teardown"]


def test_setup_requires_dataset_bundle() -> None:
    class InvalidDataPack(DataPack):
        @charge(Event.SETUP_DATA)
        def setup(self, _: DataContext) -> object:
            return object()

    with pytest.raises(TypeError, match="must return DatasetBundle"):
        DataPackHandler(InvalidDataPack()).setup(_context(InvalidDataPack()))


def test_loader_provider_requires_supported_return() -> None:
    class InvalidDataPack(DataPack):
        @charge(Event.SETUP_DATA)
        def setup(self, _: DataContext) -> DatasetBundle:
            return DatasetBundle()

        @charge(Event.CONFIGURE_DATALOADER)
        def configure(self, _: DataContext) -> object:
            return object()

    data_pack = InvalidDataPack()
    dataset = TensorDataset(torch.arange(2))
    with pytest.raises(TypeError, match="DataLoaderConfig or DataLoader"):
        DataPackHandler(data_pack).build_loader(_context(data_pack), dataset)


def test_existing_dataloader_is_returned_unchanged() -> None:
    dataset = TensorDataset(torch.arange(4))
    expected = DataLoader(dataset, batch_size=2)

    class CustomLoaderPack(DataPack):
        @charge(Event.CONFIGURE_DATALOADER)
        def configure(self, _: DataContext) -> DataLoader[Any]:
            return expected

    data_pack = CustomLoaderPack()
    actual = DataPackHandler(data_pack).build_loader(_context(data_pack), dataset)
    assert actual is expected


def test_phase_defaults_shuffle_only_map_style_training_data() -> None:
    dataset = TensorDataset(torch.arange(8))
    generator = torch.Generator().manual_seed(3)

    train_loader = materialize_dataloader(
        dataset,
        DataLoaderConfig(batch_size=2),
        phase="train",
        device=torch.device("cpu"),
        default_generator=generator,
    )
    validation_loader = materialize_dataloader(
        dataset,
        DataLoaderConfig(batch_size=2),
        phase="validation",
        device=torch.device("cpu"),
    )

    assert type(train_loader.sampler).__name__ == "RandomSampler"
    assert type(validation_loader.sampler).__name__ == "SequentialSampler"
    assert train_loader.generator is generator
    assert train_loader.pin_memory is False


def test_cuda_device_enables_automatic_pin_memory() -> None:
    loader = materialize_dataloader(
        TensorDataset(torch.arange(2)),
        DataLoaderConfig(),
        phase="test",
        device=torch.device("cuda"),
    )
    assert loader.pin_memory is True


class NumberStream(IterableDataset[int]):
    def __iter__(self):  # type: ignore[no-untyped-def]
        yield from range(4)


def test_iterable_dataset_rejects_explicit_shuffle() -> None:
    with pytest.raises(ValueError, match="IterableDataset"):
        materialize_dataloader(
            NumberStream(),
            DataLoaderConfig(shuffle=True),
            phase="train",
            device=torch.device("cpu"),
        )


def test_multiple_provider_methods_are_rejected() -> None:
    class ConflictingPack(DataPack):
        @charge(Event.SETUP_DATA)
        def first(self, _: DataContext) -> DatasetBundle:
            return DatasetBundle()

        @charge(Event.SETUP_DATA)
        def second(self, _: DataContext) -> DatasetBundle:
            return DatasetBundle()

    with pytest.raises(ValueError, match="exactly one DataPack provider"):
        DataPackHandler(ConflictingPack())


def test_data_pack_rejects_non_data_events() -> None:
    class InvalidPack(DataPack):
        @charge(Event.BEFORE_TRAIN)
        def before_train(self, _: EventContext) -> None:
            pass

    with pytest.raises(ValueError, match="cannot handle non-data event"):
        DataPackHandler(InvalidPack())


@pytest.mark.parametrize("owner", ["model", "callback"])
def test_model_and_callbacks_cannot_own_data_events(owner: str) -> None:
    class InvalidModel(nn.Module):
        @charge(Event.PREPARE_DATA)
        def prepare(self, _: DataContext) -> None:
            pass

    class InvalidCallback:
        @charge(Event.PREPARE_DATA)
        def prepare(self, _: DataContext) -> None:
            pass

    if owner == "model":
        with pytest.raises(ValueError, match=r"Model method.*DataPack event"):
            EventHandler(InvalidModel())
    else:
        with pytest.raises(ValueError, match=r"Callback.*DataPack event"):
            EventHandler(nn.Linear(1, 1), callbacks=[InvalidCallback()])
