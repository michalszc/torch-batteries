"""Base contract for event-driven data configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch_batteries.utils.logging import get_logger

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    import torch

    from .types import DataStage, ResolvedData

logger = get_logger("data.base")


class DataPack:
    """Base class for charged dataset and DataLoader configuration.

    Subclasses define data lifecycle methods with :func:`torch_batteries.charge`.
    The default checkpoint contract is stateless; subclasses may override
    :meth:`state_dict` and :meth:`load_state_dict` when dataset construction relies
    on persistent values such as split indices or streaming positions.
    """

    def resolve(
        self,
        stage: DataStage,
        *,
        device: str | torch.device = "cpu",
    ) -> AbstractContextManager[ResolvedData]:
        """Resolve datasets and DataLoaders without constructing a Battery.

        Args:
            stage: Complete workflow stage to resolve: ``"fit"``, ``"test"``, or
                ``"predict"``.
            device: PyTorch device used for device-aware loader defaults. Standalone
                resolution defaults to CPU and also accepts ``"auto"``.

        Returns:
            A context manager yielding ``ResolvedData`` and guaranteeing teardown.

        Note:
            Preparation runs once per standalone call. Keep preparation idempotent.
        """
        from .handler import DataPackHandler  # noqa: PLC0415

        return DataPackHandler(self).resolve(stage, device=device)

    def state_dict(self) -> dict[str, Any]:
        """Return state that should be stored in a full training checkpoint."""
        logger.debug("DataPack %s has no resumable state.", type(self).__name__)
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state previously returned by :meth:`state_dict`.

        Args:
            state_dict: DataPack-specific checkpoint state.
        """
        if state_dict:
            logger.warning(
                "DataPack %s ignored unexpected state keys: %s",
                type(self).__name__,
                sorted(state_dict),
            )
        logger.debug("DataPack %s state restoration completed.", type(self).__name__)
