"""DataLoader materialization for high-level DataPack configuration."""

from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset

from .types import DataLoaderConfig, DataPhase, DatasetType


def materialize_dataloader(
    dataset: DatasetType,
    config: DataLoaderConfig,
    *,
    phase: DataPhase,
    device: torch.device,
    default_generator: torch.Generator | None = None,
) -> DataLoader[Any]:
    """Construct a PyTorch DataLoader from validated high-level configuration."""
    iterable = isinstance(dataset, IterableDataset)
    shuffle = config.shuffle
    if shuffle is None:
        shuffle = phase == "train" and not iterable and config.sampler is None
    if iterable and shuffle:
        msg = "IterableDataset cannot be used with shuffle=True."
        raise ValueError(msg)

    pin_memory = (
        device.type == "cuda" if config.pin_memory == "auto" else config.pin_memory
    )
    generator = config.generator if config.generator is not None else default_generator
    common: dict[str, Any] = {
        "num_workers": config.num_workers,
        "collate_fn": config.collate_fn,
        "pin_memory": pin_memory,
        "timeout": config.timeout,
        "worker_init_fn": config.worker_init_fn,
        "generator": generator,
        "persistent_workers": config.persistent_workers,
    }
    if config.prefetch_factor is not None:
        common["prefetch_factor"] = config.prefetch_factor

    if config.batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=config.batch_sampler, **common)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=config.sampler,
        drop_last=config.drop_last,
        **common,
    )
