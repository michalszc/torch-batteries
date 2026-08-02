"""Base callback contract for torch-batteries."""

from typing import Any

from torch_batteries.utils.logging import get_logger

logger = get_logger("callbacks.base")


class Callback:
    """Base class for callbacks with optional resumable state.

    Decorator-only callback objects remain supported. Inheriting from this class
    enables a custom callback to participate in full training checkpoints.
    """

    def state_dict(self) -> dict[str, Any]:
        """Return state that should be stored in a training checkpoint."""
        logger.debug("Callback %s has no resumable state.", type(self).__name__)
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state from a training checkpoint."""
        if state_dict:
            logger.warning(
                "Callback %s ignored unexpected state keys: %s",
                type(self).__name__,
                sorted(state_dict),
            )
        logger.debug("Callback %s state restoration completed.", type(self).__name__)
