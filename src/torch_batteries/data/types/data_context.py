"""Public types used by event-driven DataPack workflows."""

from typing import TYPE_CHECKING, Any, Literal, TypedDict

import torch
from torch.utils.data import Dataset, IterableDataset

from .dataset_bundle import DatasetBundle

if TYPE_CHECKING:
    import torch_batteries
    from torch_batteries.data.base import DataPack

DatasetType = Dataset[Any] | IterableDataset[Any]

DataPhase = Literal["train", "validation", "test", "predict"]

DataStage = Literal["fit", "test", "predict"]


class DataContext(TypedDict, total=False):
    """Context passed to methods charged for DataPack lifecycle events.

    Every event receives ``data_pack``, ``stage``, and ``device``. Battery-managed
    workflows additionally receive ``battery``. ``stage`` identifies the workflow
    as ``"fit"``, ``"test"``, or ``"predict"``. A configured DataPack seed adds
    ``seed`` and a fresh generator initialized with that seed. Loader configuration
    additionally receives ``phase``, ``datasets``, ``dataset``, and ``dataset_name``.
    Teardown receives ``datasets`` only when setup succeeded.
    """

    battery: "torch_batteries.Battery"
    data_pack: "DataPack"
    stage: DataStage
    phase: DataPhase
    datasets: DatasetBundle
    dataset: DatasetType
    dataset_name: str
    device: torch.device
    seed: int
    generator: torch.Generator
