"""Public types used by event-driven DataPack workflows."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from torch.utils.data import Dataset, IterableDataset

DatasetType = Dataset[Any] | IterableDataset[Any]

DatasetCollection = DatasetType | Mapping[str, DatasetType]

DataPhase = Literal["train", "validation", "test", "predict"]


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
        """Return the dataset or named datasets configured for a workflow phase.

        Args:
            phase: ``"train"``, ``"validation"``, ``"test"``, or ``"predict"``.
        """
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
        """Return phase datasets normalized to a mapping.

        Args:
            phase: Workflow phase to normalize.
        """
        configured = self.for_phase(phase)
        if configured is None:
            return {}
        if isinstance(configured, (Dataset, IterableDataset)):
            return {"default": configured}
        return dict(configured)
