"""Internal resolved-data types used by trainer workflows."""

from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader

from torch_batteries.data.types import DataPhase


@dataclass(frozen=True, slots=True)
class ResolvedDataWorkflow:
    """Resolved loaders plus their original collection shape."""

    loaders: dict[DataPhase, dict[str, DataLoader[Any]]]
    named_phases: frozenset[DataPhase]
