"""Batch scheduling across named phase loaders."""

import random
from collections.abc import Iterator, Mapping
from typing import Any

from torch.utils.data import DataLoader

from torch_batteries.data.types.batch_schedule import BatchScheduleConfig


def scheduled_batches(
    loaders: Mapping[str, DataLoader[Any]],
    schedule: BatchScheduleConfig,
    epoch: int,
) -> Iterator[tuple[str, Any]]:
    """Yield every batch once using the configured deterministic schedule.

    Args:
        loaders: Named loaders to exhaust.
        schedule: Batch ordering and random seed.
        epoch: Absolute epoch used to derive interleave order.

    Yields:
        Dataset name and next batch.
    """
    remaining = {name: len(loader) for name, loader in loaders.items()}
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    names = list(loaders)
    rng = random.Random(schedule.seed + epoch)
    position = 0
    while any(remaining.values()):
        if schedule.mode == "round_robin":
            name = names[position % len(names)]
            position += 1
            if remaining[name] == 0:
                continue
        else:
            active = [name for name in names if remaining[name] > 0]
            name = rng.choices(active, weights=[remaining[item] for item in active])[0]
        remaining[name] -= 1
        yield name, next(iterators[name])
