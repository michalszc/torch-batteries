"""Public types used by event-driven DataPack workflows."""

from dataclasses import dataclass
from typing import Literal

import torch

from .data_loader_bundle import DataLoaderBundle
from .dataset_bundle import DatasetBundle

DataStage = Literal["fit", "test", "predict"]


@dataclass(frozen=True, slots=True)
class ResolvedData:
    """Datasets and DataLoaders materialized for one DataPack stage.

    Args:
        stage: Workflow stage resolved by the DataPack.
        device: Device used when configuring loaders.
        datasets: Datasets returned by the DataPack's setup handler.
        loaders: DataLoaders built for the selected stage.
    """

    stage: DataStage
    device: torch.device
    datasets: DatasetBundle
    loaders: DataLoaderBundle
