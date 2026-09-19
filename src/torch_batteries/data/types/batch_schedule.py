"""Scheduling of batches from named datasets."""

from dataclasses import dataclass
from typing import Literal

BatchScheduleMode = Literal["round_robin", "interleave"]


@dataclass(frozen=True, slots=True)
class BatchScheduleConfig:
    """Select a per-phase batch order for named datasets.

    ``interleave`` samples a loader in proportion to its remaining batches.
    ``seed`` makes its order reproducible for each absolute epoch.
    """

    mode: BatchScheduleMode = "round_robin"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"round_robin", "interleave"}:
            msg = "Batch schedule mode must be 'round_robin' or 'interleave'."
            raise ValueError(msg)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            msg = "Batch schedule seed must be an integer."
            raise TypeError(msg)
