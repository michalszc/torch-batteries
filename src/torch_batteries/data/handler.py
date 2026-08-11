"""Charged DataPack discovery and DataLoader materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from torch.utils.data import DataLoader

from torch_batteries.events import Event
from torch_batteries.utils.logging import get_logger

from .loader import materialize_dataloader
from .types import DataContext, DataLoaderConfig, DatasetBundle, DatasetType

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base import DataPack

logger = get_logger("data.handler")


class DataPackHandler:
    """Discover and dispatch lifecycle methods charged on one DataPack."""

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

    def __init__(self, data_pack: DataPack) -> None:
        self.data_pack = data_pack
        self._handlers: dict[Event, list[Callable[[DataContext], Any]]] = {}
        self._labels: dict[Event, list[str]] = {}
        self._discover_handlers()
        self._validate_providers()

    def _discover_handlers(self) -> None:
        """Discover charged DataPack methods and reject unrelated events."""
        for name in dir(self.data_pack):
            method = getattr(self.data_pack, name)
            if not callable(method) or not hasattr(method, "_torch_batteries_event"):
                continue
            event = method._torch_batteries_event  # noqa: SLF001
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
        """Return whether the DataPack handles an event."""
        return bool(self._handlers.get(event))

    def call(self, event: Event, context: DataContext) -> None:
        """Call all ordered handlers for a side-effect data event."""
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
        """Return a provider result or the supplied default."""
        if event not in self.PROVIDER_EVENTS:
            msg = f"Event '{event.value}' is not a DataPack provider event."
            raise ValueError(msg)
        handlers = self._handlers.get(event, [])
        return default if not handlers else handlers[0](context)

    def setup(self, context: DataContext) -> DatasetBundle:
        """Construct and validate the datasets for one workflow invocation."""
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
        """Resolve a custom loader or materialize a DataLoaderConfig."""
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
