"""Charged DataPack discovery and DataLoader materialization."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.utils.data import DataLoader

from torch_batteries.events import Event
from torch_batteries.events._metadata import get_charged_events
from torch_batteries.utils.device import get_device
from torch_batteries.utils.logging import get_logger

from .loader import materialize_dataloader
from .types import (
    DataContext,
    DataLoaderBundle,
    DataLoaderCollection,
    DataLoaderConfig,
    DataPhase,
    DatasetBundle,
    DatasetType,
    DataStage,
    ResolvedData,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch_batteries

    from .base import DataPack

logger = get_logger("data.handler")


class DataPackHandler:
    """Discover and dispatch lifecycle methods charged on one DataPack.

    Args:
        data_pack: DataPack whose charged lifecycle methods are discovered.
    """

    DATA_EVENTS: ClassVar[set[Event]] = {
        Event.PREPARE_DATA,
        Event.SETUP_DATA,
        Event.CONFIGURE_DATALOADER,
        Event.TEARDOWN_DATA,
    }
    PROVIDER_EVENTS: ClassVar[set[Event]] = {
        Event.SETUP_DATA,
        Event.CONFIGURE_DATALOADER,
    }
    STAGE_PHASES: ClassVar[dict[DataStage, tuple[tuple[DataPhase, bool], ...]]] = {
        "fit": (("train", True), ("validation", False)),
        "test": (("test", True),),
        "predict": (("predict", True),),
    }

    def __init__(self, data_pack: DataPack) -> None:
        self.data_pack = data_pack
        self._handlers: dict[Event, list[Callable[[DataContext], Any]]] = {}
        self._labels: dict[Event, list[str]] = {}
        self._discover_handlers()
        self._validate_providers()
        self._prepared = False
        logger.debug(
            "DataPack handler initialized: data_pack=%s, events=%s",
            type(data_pack).__name__,
            sorted(event.value for event in self._handlers),
        )

    def _discover_handlers(self) -> None:
        """Discover charged DataPack methods and reject unrelated events."""
        for name in dir(self.data_pack):
            method = getattr(self.data_pack, name)
            if not callable(method):
                continue
            events = get_charged_events(method)
            if len(events) != len(set(events)):
                msg = (
                    f"DataPack '{type(self.data_pack).__name__}' method '{name}' "
                    "is charged repeatedly for one event."
                )
                raise ValueError(msg)
            for event in events:
                if event not in self.DATA_EVENTS:
                    msg = (
                        f"DataPack '{type(self.data_pack).__name__}' method '{name}' "
                        f"cannot handle non-data event '{event.value}'."
                    )
                    raise ValueError(msg)
                self._handlers.setdefault(event, []).append(method)
                self._labels.setdefault(event, []).append(
                    f"{type(self.data_pack).__name__}.{name}"
                )

    def _validate_providers(self) -> None:
        """Require at most one owner for each data provider event."""
        for event in self.PROVIDER_EVENTS:
            labels = self._labels.get(event, [])
            if len(labels) > 1:
                joined = ", ".join(labels)
                msg = (
                    f"Event '{event.value}' accepts exactly one DataPack provider; "
                    f"found: {joined}."
                )
                raise ValueError(msg)

    def has_handler(self, event: Event) -> bool:
        """Return whether the DataPack handles an event.

        Args:
            event: Data lifecycle event to inspect.
        """
        return bool(self._handlers.get(event))

    def call(self, event: Event, context: DataContext) -> None:
        """Call all ordered handlers for a side-effect data event.

        Args:
            event: Side-effect event to dispatch.
            context: Data lifecycle context passed to handlers.
        """
        if event not in self.DATA_EVENTS - self.PROVIDER_EVENTS:
            msg = f"Event '{event.value}' is not a DataPack side-effect event."
            raise ValueError(msg)
        for handler in self._handlers.get(event, []):
            result = handler(context)
            if result is not None:
                msg = f"Event '{event.value}' handlers must return None."
                raise TypeError(msg)

    def provide(
        self,
        event: Event,
        context: DataContext,
        *,
        default: Any,
    ) -> Any:
        """Return a provider result or the supplied default.

        Args:
            event: Provider event to dispatch.
            context: Data lifecycle context passed to the provider.
            default: Value returned when no provider is registered.
        """
        if event not in self.PROVIDER_EVENTS:
            msg = f"Event '{event.value}' is not a DataPack provider event."
            raise ValueError(msg)
        handlers = self._handlers.get(event, [])
        return default if not handlers else handlers[0](context)

    def setup(self, context: DataContext) -> DatasetBundle:
        """Construct and validate datasets for one workflow invocation.

        Args:
            context: Setup context passed to the dataset provider.
        """
        result = self.provide(Event.SETUP_DATA, context, default=None)
        if not isinstance(result, DatasetBundle):
            returned = type(result).__name__
            msg = f"SETUP_DATA handler must return DatasetBundle, got {returned}."
            raise TypeError(msg)
        return result

    def build_loader(
        self,
        context: DataContext,
        dataset: DatasetType,
    ) -> DataLoader[Any]:
        """Resolve a custom loader or materialize a DataLoaderConfig.

        Args:
            context: Loader configuration context.
            dataset: Dataset for which a loader is required.
        """
        result = self.provide(
            Event.CONFIGURE_DATALOADER,
            context,
            default=DataLoaderConfig(),
        )
        if isinstance(result, DataLoader):
            return result
        if not isinstance(result, DataLoaderConfig):
            returned = type(result).__name__
            msg = (
                "CONFIGURE_DATALOADER handler must return DataLoaderConfig or "
                f"DataLoader, got {returned}."
            )
            raise TypeError(msg)
        return materialize_dataloader(
            dataset,
            result,
            phase=context["phase"],
            device=context["device"],
            default_generator=context.get("generator"),
        )

    def _data_seed(self) -> int | None:
        """Return the optional validated seed exposed by the DataPack."""
        seed: object = getattr(self.data_pack, "seed", None)
        if seed is None:
            return None
        if isinstance(seed, bool) or not isinstance(seed, int):
            msg = "DataPack seed must be an integer or None."
            raise TypeError(msg)
        if seed < 0:
            msg = "DataPack seed must be a non-negative integer."
            raise ValueError(msg)
        return seed

    def _context(  # noqa: PLR0913
        self,
        stage: DataStage,
        device: torch.device,
        *,
        battery: torch_batteries.Battery | None,
        phase: DataPhase | None = None,
        datasets: DatasetBundle | None = None,
        dataset_name: str | None = None,
    ) -> DataContext:
        """Build a deterministic context for one DataPack lifecycle event."""
        seed = self._data_seed()
        context: DataContext = {
            "data_pack": self.data_pack,
            "stage": stage,
            "device": device,
        }
        if battery is not None:
            context["battery"] = battery
        if seed is not None:
            context["seed"] = seed
            context["generator"] = torch.Generator().manual_seed(seed)
        if phase is not None:
            context["phase"] = phase
        if datasets is not None:
            context["datasets"] = datasets
        if dataset_name is not None:
            context["dataset_name"] = dataset_name
        return context

    @contextmanager
    def resolve(  # noqa: PLR0912, PLR0915
        self,
        stage: DataStage,
        *,
        device: str | torch.device = "cpu",
        battery: torch_batteries.Battery | None = None,
        dataset_name: str | None = None,
    ) -> Generator[ResolvedData]:
        """Resolve one DataPack stage and guarantee workflow teardown.

        Args:
            stage: ``"fit"``, ``"test"``, or ``"predict"``.
            device: Device used for loader configuration.
            battery: Optional owning Battery included in event contexts.
            dataset_name: Optional named test or prediction dataset selection.

        Yields:
            Resolved datasets and loaders for the requested stage.
        """
        resolved_device = get_device(device)
        logger.debug(
            "DataPack resolution started: data_pack=%s, stage=%s, device=%s",
            type(self.data_pack).__name__,
            stage,
            resolved_device,
        )
        setup_context = self._context(stage, resolved_device, battery=battery)
        datasets: DatasetBundle | None = None
        try:
            if not self._prepared:
                self.call(Event.PREPARE_DATA, setup_context)
                self._prepared = True
            datasets = self.setup(setup_context)

            train_loader: DataLoader[Any] | None = None
            validation_loader: DataLoader[Any] | None = None
            test_loaders: DataLoader[Any] | Mapping[str, DataLoader[Any]] | None = None
            predict_loaders: DataLoader[Any] | Mapping[str, DataLoader[Any]] | None = (
                None
            )

            for phase, required in self.STAGE_PHASES[stage]:
                configured = datasets.for_phase(phase)
                is_named = isinstance(configured, Mapping)
                phase_datasets = datasets.datasets_for_phase(phase)
                if not phase_datasets:
                    if required:
                        msg = (
                            f"DataPack '{type(self.data_pack).__name__}' did not "
                            f"provide a dataset for phase '{phase}'."
                        )
                        raise ValueError(msg)
                    continue
                if dataset_name is not None:
                    if dataset_name not in phase_datasets:
                        available = ", ".join(repr(name) for name in phase_datasets)
                        msg = (
                            f"Unknown dataset {dataset_name!r} for phase '{phase}'. "
                            f"Available datasets: {available}."
                        )
                        raise ValueError(msg)
                    phase_datasets = {dataset_name: phase_datasets[dataset_name]}

                phase_loaders: dict[str, DataLoader[Any]] = {}
                for name, dataset in phase_datasets.items():
                    context = self._context(
                        stage,
                        resolved_device,
                        battery=battery,
                        phase=phase,
                        datasets=datasets,
                        dataset_name=name,
                    )
                    context["dataset"] = dataset
                    phase_loaders[name] = self.build_loader(context, dataset)

                resolved_loaders: DataLoaderCollection
                if is_named:
                    resolved_loaders = phase_loaders
                else:
                    resolved_loaders = next(iter(phase_loaders.values()))
                match phase:
                    case "train":
                        assert isinstance(resolved_loaders, DataLoader)
                        train_loader = resolved_loaders
                    case "validation":
                        assert isinstance(resolved_loaders, DataLoader)
                        validation_loader = resolved_loaders
                    case "test":
                        test_loaders = resolved_loaders
                    case "predict":
                        predict_loaders = resolved_loaders

            yield ResolvedData(
                stage=stage,
                device=resolved_device,
                datasets=datasets,
                loaders=DataLoaderBundle(
                    train=train_loader,
                    validation=validation_loader,
                    test=test_loaders,
                    predict=predict_loaders,
                ),
            )
        finally:
            teardown_context = self._context(
                stage,
                resolved_device,
                battery=battery,
                datasets=datasets,
            )
            self.call(Event.TEARDOWN_DATA, teardown_context)
            logger.debug(
                "DataPack teardown completed: data_pack=%s, stage=%s",
                type(self.data_pack).__name__,
                stage,
            )
