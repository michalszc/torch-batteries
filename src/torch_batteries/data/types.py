"""Public types used by event-driven DataPack workflows."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import torch
from torch.utils.data import Dataset, IterableDataset, Sampler

if TYPE_CHECKING:
    import torch_batteries

    from .base import DataPack

DatasetType = Dataset[Any] | IterableDataset[Any]
DatasetCollection = DatasetType | Mapping[str, DatasetType]
DataPhase = Literal["train", "validation", "test", "predict"]
DataStage = Literal["fit", "test", "predict"]


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Datasets made available by a charged ``SETUP_DATA`` provider.

    Training and validation accept one PyTorch dataset. Test and prediction also
    accept a non-empty mapping of non-blank names to PyTorch datasets.
    """

    train: DatasetType | None = None
    validation: DatasetType | None = None
    test: DatasetCollection | None = None
    predict: DatasetCollection | None = None

    def __post_init__(self) -> None:
        """Validate every configured dataset against its phase contract."""
        for phase in ("train", "validation"):
            configured = getattr(self, phase)
            if configured is None or isinstance(configured, (Dataset, IterableDataset)):
                continue
            returned = type(configured).__name__
            msg = (
                f"DatasetBundle {phase} dataset must be a PyTorch Dataset or "
                f"IterableDataset, or None, got {returned}."
            )
            raise TypeError(msg)

        for phase in ("test", "predict"):
            configured = getattr(self, phase)
            if configured is None or isinstance(configured, (Dataset, IterableDataset)):
                continue
            if not isinstance(configured, Mapping):
                returned = type(configured).__name__
                msg = (
                    f"DatasetBundle {phase} dataset must be a PyTorch Dataset or "
                    "IterableDataset, a non-empty mapping of dataset names to "
                    f"datasets, or None, got {returned}."
                )
                raise TypeError(msg)
            if not configured:
                msg = f"DatasetBundle {phase} dataset mapping cannot be empty."
                raise ValueError(msg)
            for name, dataset in configured.items():
                if not isinstance(name, str) or not name.strip():
                    msg = (
                        f"DatasetBundle {phase} dataset names must be non-blank "
                        "strings."
                    )
                    raise ValueError(msg)
                if not isinstance(dataset, (Dataset, IterableDataset)):
                    returned = type(dataset).__name__
                    msg = (
                        f"DatasetBundle {phase} dataset '{name}' must be a PyTorch "
                        f"Dataset or IterableDataset, got {returned}."
                    )
                    raise TypeError(msg)

    def for_phase(self, phase: DataPhase) -> DatasetCollection | None:
        """Return the dataset or named datasets configured for a workflow phase."""
        match phase:
            case "train":
                return self.train
            case "validation":
                return self.validation
            case "test":
                return self.test
            case "predict":
                return self.predict

    def datasets_for_phase(self, phase: DataPhase) -> dict[str, DatasetType]:
        """Return phase datasets normalized to a mapping."""
        configured = self.for_phase(phase)
        if configured is None:
            return {}
        if isinstance(configured, (Dataset, IterableDataset)):
            return {"default": configured}
        return dict(configured)


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

    def __post_init__(self) -> None:
        """Reject combinations that PyTorch cannot materialize safely."""
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool) or self.batch_size < 1
        ):
            msg = "DataLoaderConfig batch_size must be positive or None."
            raise ValueError(msg)
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            msg = "DataLoaderConfig num_workers must be a non-negative integer."
            raise ValueError(msg)
        if self.timeout < 0:
            msg = "DataLoaderConfig timeout must be non-negative."
            raise ValueError(msg)
        if self.prefetch_factor is not None and self.prefetch_factor < 1:
            msg = "DataLoaderConfig prefetch_factor must be positive when provided."
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


class DataContext(TypedDict, total=False):
    """Context passed to methods charged for DataPack lifecycle events."""

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
