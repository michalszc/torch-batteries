"""Public types used by event-driven DataPack workflows."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from torch.utils.data import DataLoader

DataLoaderCollection = DataLoader[Any] | Mapping[str, DataLoader[Any]]

DataPhase = Literal["train", "validation", "test", "predict"]


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
        """Return the loader or named loaders configured for a workflow phase.

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

    def loaders_for_phase(self, phase: DataPhase) -> dict[str, DataLoader[Any]]:
        """Return phase loaders normalized to a mapping.

        Args:
            phase: Workflow phase to normalize.
        """
        configured = self.for_phase(phase)
        if configured is None:
            return {}
        if isinstance(configured, DataLoader):
            return {"default": configured}
        return dict(configured)
