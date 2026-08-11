"""Base contract for event-driven data configuration."""

from typing import Any

from torch_batteries.utils.logging import get_logger

logger = get_logger("data.base")


class DataPack:
    """Base class for charged dataset and DataLoader configuration.

    Subclasses define data lifecycle methods with :func:`torch_batteries.charge`.
    The default checkpoint contract is stateless; subclasses may override
    :meth:`state_dict` and :meth:`load_state_dict` when dataset construction relies
    on persistent values such as split indices or streaming positions.
    """

    def state_dict(self) -> dict[str, Any]:
        """Return state that should be stored in a full training checkpoint."""
        logger.debug("DataPack %s has no resumable state.", type(self).__name__)
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state previously returned by :meth:`state_dict`."""
        if state_dict:
            logger.warning(
                "DataPack %s ignored unexpected state keys: %s",
                type(self).__name__,
                sorted(state_dict),
            )
        logger.debug("DataPack %s state restoration completed.", type(self).__name__)
