"""Public types used by event-driven DataPack workflows."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

if TYPE_CHECKING:
    import torch_batteries

    from .base import DataPack

DatasetType = Dataset[Any] | IterableDataset[Any]
DatasetCollection = DatasetType | Mapping[str, DatasetType]
DataLoaderCollection = DataLoader[Any] | Mapping[str, DataLoader[Any]]
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
class DataLoaderBundle:
    """DataLoaders resolved for one DataPack stage.

    Training and validation contain at most one loader. Test and prediction retain
    whether their datasets were configured as a bare value or a named mapping.
    """

    train: DataLoader[Any] | None = None
    validation: DataLoader[Any] | None = None
    test: DataLoaderCollection | None = None
    predict: DataLoaderCollection | None = None

    def __post_init__(self) -> None:
        """Validate every configured loader against its phase contract."""
        for phase in ("train", "validation"):
            configured = getattr(self, phase)
            if configured is None or isinstance(configured, DataLoader):
                continue
            returned = type(configured).__name__
            msg = (
                f"DataLoaderBundle {phase} loader must be a DataLoader or None, "
                f"got {returned}."
            )
            raise TypeError(msg)

        for phase in ("test", "predict"):
            configured = getattr(self, phase)
            if configured is None or isinstance(configured, DataLoader):
                continue
            if not isinstance(configured, Mapping):
                returned = type(configured).__name__
                msg = (
                    f"DataLoaderBundle {phase} loader must be a DataLoader, a "
                    "non-empty mapping of loader names to DataLoaders, or None, "
                    f"got {returned}."
                )
                raise TypeError(msg)
            if not configured:
                msg = f"DataLoaderBundle {phase} loader mapping cannot be empty."
                raise ValueError(msg)
            for name, loader in configured.items():
                if not isinstance(name, str) or not name.strip():
                    msg = (
                        f"DataLoaderBundle {phase} loader names must be non-blank "
                        "strings."
                    )
                    raise ValueError(msg)
                if not isinstance(loader, DataLoader):
                    returned = type(loader).__name__
                    msg = (
                        f"DataLoaderBundle {phase} loader '{name}' must be a "
                        f"DataLoader, got {returned}."
                    )
                    raise TypeError(msg)

    def for_phase(self, phase: DataPhase) -> DataLoaderCollection | None:
        """Return the loader or named loaders configured for a workflow phase."""
        match phase:
            case "train":
                return self.train
            case "validation":
                return self.validation
            case "test":
                return self.test
            case "predict":
                return self.predict

    def loaders_for_phase(self, phase: DataPhase) -> dict[str, DataLoader[Any]]:
        """Return phase loaders normalized to a mapping."""
        configured = self.for_phase(phase)
        if configured is None:
            return {}
        if isinstance(configured, DataLoader):
            return {"default": configured}
        return dict(configured)


@dataclass(frozen=True, slots=True)
class ResolvedData:
    """Datasets and DataLoaders materialized for one DataPack stage."""

    stage: DataStage
    device: torch.device
    datasets: DatasetBundle
    loaders: DataLoaderBundle


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
