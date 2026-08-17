"""Tests for charged DataPack dispatch and DataLoader construction."""

from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    IterableDataset,
    SequentialSampler,
    TensorDataset,
)

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


def test_handler_reports_registered_events() -> None:
    handler = DataPackHandler(ExampleDataPack())

    assert handler.has_handler(Event.SETUP_DATA)
    assert not handler.has_handler(Event.BEFORE_TRAIN)


def test_call_rejects_provider_event() -> None:
    data_pack = ExampleDataPack()
    handler = DataPackHandler(data_pack)

    with pytest.raises(ValueError, match="not a DataPack side-effect event"):
        handler.call(Event.SETUP_DATA, _context(data_pack))


def test_provide_rejects_side_effect_event() -> None:
    data_pack = ExampleDataPack()
    handler = DataPackHandler(data_pack)

    with pytest.raises(ValueError, match="not a DataPack provider event"):
        handler.provide(Event.PREPARE_DATA, _context(data_pack), default=None)


def test_side_effect_handler_must_return_none() -> None:
    class InvalidDataPack(DataPack):
        @charge(Event.PREPARE_DATA)
        def prepare(self, _: DataContext) -> str:
            return "invalid"

    data_pack = InvalidDataPack()

    with pytest.raises(TypeError, match="handlers must return None"):
        DataPackHandler(data_pack).call(Event.PREPARE_DATA, _context(data_pack))


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


def test_data_pack_resolves_fit_without_a_battery() -> None:
    data_pack = ExampleDataPack()

    with data_pack.resolve("fit") as resolved:
        assert resolved.stage == "fit"
        assert resolved.device == torch.device("cpu")
        assert resolved.datasets.train is data_pack.dataset
        assert resolved.loaders.train is not None
        assert len(resolved.loaders.train) == 3
        assert resolved.loaders.validation is None
        assert data_pack.calls == ["prepare", "setup", "loader:train"]

    assert data_pack.calls == ["prepare", "setup", "loader:train", "teardown"]


@pytest.mark.parametrize("stage", ["test", "predict"])
def test_data_pack_preserves_named_loader_shape(stage: str) -> None:
    first = TensorDataset(torch.arange(2))
    second = TensorDataset(torch.arange(3))

    class NamedDataPack(DataPack):
        @charge(Event.SETUP_DATA)
        def setup(self, context: DataContext) -> DatasetBundle:
            named = {"first": first, "second": second}
            if context["stage"] == "test":
                return DatasetBundle(test=named)
            return DatasetBundle(predict=named)

    with NamedDataPack().resolve(stage) as resolved:  # type: ignore[arg-type]
        phase_loaders = resolved.loaders.for_phase(stage)  # type: ignore[arg-type]
        assert isinstance(phase_loaders, dict)
        assert list(phase_loaders) == ["first", "second"]
        assert resolved.loaders.loaders_for_phase(stage) == phase_loaders  # type: ignore[arg-type]


def test_resolve_uses_default_loader_configuration_and_auto_device() -> None:
    class DefaultDataPack(DataPack):
        @charge(Event.SETUP_DATA)
        def setup(self, _: DataContext) -> DatasetBundle:
            return DatasetBundle(train=TensorDataset(torch.arange(2)))

    with DefaultDataPack().resolve("fit", device="auto") as resolved:
        loader = resolved.loaders.train
        assert loader is not None
        assert loader.batch_size == 1
        assert isinstance(resolved.device, torch.device)


def test_standalone_resolution_context_omits_battery_and_reuses_seed() -> None:
    class ContextDataPack(DataPack):
        seed = 7

        def __init__(self) -> None:
            self.contexts: list[DataContext] = []
            self.dataset = TensorDataset(torch.arange(4))

        @charge(Event.SETUP_DATA)
        def setup(self, context: DataContext) -> DatasetBundle:
            self.contexts.append(context.copy())
            return DatasetBundle(train=self.dataset, validation=self.dataset)

        @charge(Event.CONFIGURE_DATALOADER)
        def configure(self, context: DataContext) -> DataLoaderConfig:
            self.contexts.append(context.copy())
            return DataLoaderConfig()

        @charge(Event.TEARDOWN_DATA)
        def teardown(self, context: DataContext) -> None:
            self.contexts.append(context.copy())

    data_pack = ContextDataPack()
    with data_pack.resolve("fit"):
        pass

    assert all("battery" not in context for context in data_pack.contexts)
    assert data_pack.contexts[0]["generator"].initial_seed() == 7
    loader_contexts = [context for context in data_pack.contexts if "phase" in context]
    assert [context["phase"] for context in loader_contexts] == [
        "train",
        "validation",
    ]
    assert [context["generator"].initial_seed() for context in loader_contexts] == [
        7,
        7,
    ]
    assert data_pack.contexts[-1]["datasets"].train is data_pack.dataset


def test_reused_handler_prepares_once_but_public_resolve_prepares_per_call() -> None:
    data_pack = ExampleDataPack()
    handler = DataPackHandler(data_pack)

    with handler.resolve("fit"):
        pass
    with handler.resolve("fit"):
        pass
    assert data_pack.calls.count("prepare") == 1

    separate_data_pack = ExampleDataPack()
    with separate_data_pack.resolve("fit"):
        pass
    with separate_data_pack.resolve("fit"):
        pass
    assert separate_data_pack.calls.count("prepare") == 2


@pytest.mark.parametrize(
    ("stage", "phase"),
    [("fit", "train"), ("test", "test"), ("predict", "predict")],
)
def test_resolve_requires_the_primary_stage_dataset(stage: str, phase: str) -> None:
    class EmptyDataPack(DataPack):
        @charge(Event.SETUP_DATA)
        def setup(self, _: DataContext) -> DatasetBundle:
            return DatasetBundle()

    with (
        pytest.raises(ValueError, match=f"dataset for phase '{phase}'"),
        EmptyDataPack().resolve(stage),  # type: ignore[arg-type]
    ):
        pass


@pytest.mark.parametrize("seed", [True, -1, 1.5, "7"])
def test_standalone_resolve_rejects_invalid_seed(seed: object) -> None:
    class InvalidSeedDataPack(ExampleDataPack):
        seed: object = None

    data_pack = InvalidSeedDataPack()
    data_pack.seed = seed

    with (
        pytest.raises(ValueError, match="non-negative integer"),
        data_pack.resolve("fit"),
    ):
        pass


def test_handler_resolve_supports_dataset_selection() -> None:
    first = TensorDataset(torch.arange(2))
    second = TensorDataset(torch.arange(3))

    class NamedDataPack(DataPack):
        @charge(Event.SETUP_DATA)
        def setup(self, _: DataContext) -> DatasetBundle:
            return DatasetBundle(predict={"first": first, "second": second})

    handler = DataPackHandler(NamedDataPack())
    with (
        pytest.raises(ValueError, match="Unknown dataset 'missing'"),
        handler.resolve("predict", dataset_name="missing"),
    ):
        pass

    with handler.resolve("predict", dataset_name="second") as resolved:
        loaders = resolved.loaders.loaders_for_phase("predict")
        assert list(loaders) == ["second"]


@pytest.mark.parametrize("failure", ["prepare", "setup", "configure", "body"])
def test_resolve_tears_down_after_lifecycle_and_body_failures(failure: str) -> None:
    class FailingDataPack(DataPack):
        def __init__(self) -> None:
            self.calls: list[str] = []

        @charge(Event.PREPARE_DATA)
        def prepare(self, _: DataContext) -> None:
            self.calls.append("prepare")
            if failure == "prepare":
                msg = "prepare failed"
                raise RuntimeError(msg)

        @charge(Event.SETUP_DATA)
        def setup(self, _: DataContext) -> DatasetBundle:
            self.calls.append("setup")
            if failure == "setup":
                msg = "setup failed"
                raise RuntimeError(msg)
            return DatasetBundle(train=TensorDataset(torch.arange(2)))

        @charge(Event.CONFIGURE_DATALOADER)
        def configure(self, _: DataContext) -> DataLoaderConfig:
            self.calls.append("configure")
            if failure == "configure":
                msg = "configure failed"
                raise RuntimeError(msg)
            return DataLoaderConfig()

        @charge(Event.TEARDOWN_DATA)
        def teardown(self, _: DataContext) -> None:
            self.calls.append("teardown")

    def run_resolution(data_pack: FailingDataPack) -> None:
        with data_pack.resolve("fit"):
            if failure == "body":
                msg = "body failed"
                raise RuntimeError(msg)

    data_pack = FailingDataPack()
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        run_resolution(data_pack)

    assert data_pack.calls[-1] == "teardown"


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


def test_worker_prefetch_configuration_is_materialized() -> None:
    loader = materialize_dataloader(
        TensorDataset(torch.arange(4)),
        DataLoaderConfig(num_workers=1, prefetch_factor=3),
        phase="test",
        device=torch.device("cpu"),
    )

    assert loader.num_workers == 1
    assert loader.prefetch_factor == 3


def test_batch_sampler_configuration_is_materialized() -> None:
    dataset = TensorDataset(torch.arange(4))
    batch_sampler = BatchSampler(
        SequentialSampler(dataset), batch_size=2, drop_last=False
    )

    loader = materialize_dataloader(
        dataset,
        DataLoaderConfig(batch_size=None, batch_sampler=batch_sampler),
        phase="train",
        device=torch.device("cpu"),
    )

    assert loader.batch_sampler is batch_sampler


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
