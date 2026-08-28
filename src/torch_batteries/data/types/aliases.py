"""Shared aliases for DataPack contracts."""

from collections.abc import Mapping
from typing import Any, Literal

from torch.utils.data import DataLoader, Dataset, IterableDataset

DatasetType = Dataset[Any] | IterableDataset[Any]
DatasetCollection = DatasetType | Mapping[str, DatasetType]
DataLoaderCollection = DataLoader[Any] | Mapping[str, DataLoader[Any]]
DataPhase = Literal["train", "validation", "test", "predict"]
DataStage = Literal["fit", "test", "predict"]
