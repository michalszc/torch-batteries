"""Public types used by event-driven DataPack workflows."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    """Validated high-level configuration used to construct a DataLoader.

    ``shuffle=None`` selects the phase default and ``pin_memory="auto"`` lets the
    runtime select pinning from the Battery device. Setting ``batch_sampler`` requires
    ``batch_size=None`` and conflicts with shuffle, sampler, and drop-last options.
    """

    batch_size: int | None = 1
    shuffle: bool | None = None
    sampler: Sampler[Any] | Iterable[Any] | None = None
    batch_sampler: Sampler[list[int]] | Iterable[list[int]] | None = None
    num_workers: int = 0
    collate_fn: Callable[[list[Any]], Any] | None = None
    pin_memory: bool | Literal["auto"] = "auto"
    drop_last: bool = False
    timeout: float = 0
    worker_init_fn: Callable[[int], None] | None = None
    generator: torch.Generator | None = None
    prefetch_factor: int | None = None
    persistent_workers: bool = False

    def __post_init__(self) -> None:  # noqa: PLR0912, PLR0915
        """Reject types and combinations PyTorch cannot materialize safely."""
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int)
        ):
            msg = "DataLoaderConfig batch_size must be an integer or None."
            raise TypeError(msg)
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int):
            msg = "DataLoaderConfig num_workers must be an integer."
            raise TypeError(msg)
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, int | float):
            msg = "DataLoaderConfig timeout must be numeric."
            raise TypeError(msg)
        if self.prefetch_factor is not None and (
            isinstance(self.prefetch_factor, bool)
            or not isinstance(self.prefetch_factor, int)
        ):
            msg = "DataLoaderConfig prefetch_factor must be an integer or None."
            raise TypeError(msg)
        if type(cast("object", self.shuffle)) not in {bool, type(None)}:
            msg = "DataLoaderConfig shuffle must be a boolean or None."
            raise TypeError(msg)
        if type(cast("object", self.drop_last)) is not bool:
            msg = "DataLoaderConfig drop_last must be a boolean."
            raise TypeError(msg)
        if type(cast("object", self.pin_memory)) not in {bool, str}:
            msg = "DataLoaderConfig pin_memory must be a boolean or 'auto'."
            raise TypeError(msg)
        if type(cast("object", self.persistent_workers)) is not bool:
            msg = "DataLoaderConfig persistent_workers must be a boolean."
            raise TypeError(msg)
        if self.batch_size is not None and self.batch_size < 1:
            msg = "DataLoaderConfig batch_size must be positive or None."
            raise ValueError(msg)
        if self.num_workers < 0:
            msg = "DataLoaderConfig num_workers must be a non-negative integer."
            raise ValueError(msg)
        if self.timeout < 0:
            msg = "DataLoaderConfig timeout must be non-negative."
            raise ValueError(msg)
        if self.prefetch_factor is not None and self.prefetch_factor < 1:
            msg = "DataLoaderConfig prefetch_factor must be positive when provided."
            raise ValueError(msg)
        if self.pin_memory not in {True, False, "auto"}:
            msg = "DataLoaderConfig pin_memory must be a boolean or 'auto'."
            raise ValueError(msg)
        if self.persistent_workers and self.num_workers == 0:
            msg = "persistent_workers requires num_workers greater than zero."
            raise ValueError(msg)
        if self.prefetch_factor is not None and self.num_workers == 0:
            msg = "prefetch_factor requires num_workers greater than zero."
            raise ValueError(msg)
        if self.sampler is not None and self.shuffle is True:
            msg = "sampler and shuffle=True are mutually exclusive."
            raise ValueError(msg)
        if self.batch_sampler is not None and (
            self.batch_size is not None
            or self.shuffle not in {None, False}
            or self.sampler is not None
            or self.drop_last
        ):
            msg = (
                "batch_sampler requires batch_size=None and cannot be combined with "
                "shuffle, sampler, or drop_last."
            )
            raise ValueError(msg)
