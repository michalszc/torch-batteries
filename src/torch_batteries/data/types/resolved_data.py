"""Public types used by event-driven DataPack workflows."""

from dataclasses import dataclass
from typing import Literal

import torch

from .data_loader_bundle import DataLoaderBundle
from .dataset_bundle import DatasetBundle

DataStage = Literal["fit", "test", "predict"]


@dataclass(frozen=True, slots=True)
class ResolvedData:
    """Datasets and DataLoaders materialized for one DataPack stage."""

    stage: DataStage
    device: torch.device
    datasets: DatasetBundle
    loaders: DataLoaderBundle
