"""Gradient accumulation training control."""

from typing import Any

from torch_batteries.callbacks.base import Callback
from torch_batteries.utils.logging import get_logger

logger = get_logger("GradientAccumulation")


class GradientAccumulation(Callback):
    """Accumulate gradients across multiple batches before optimizer steps.

    Args:
        steps: Maximum number of batches in one accumulation group.
    """

    __slots__ = ("_optimizer_step_idx", "_steps")

    def __init__(self, steps: int) -> None:
        if steps < 1:
            logger.error("Invalid gradient accumulation steps: %d", steps)
            msg = "GradientAccumulation steps must be greater than zero."
            raise ValueError(msg)
        self._steps = steps
        self._optimizer_step_idx = 0
        logger.info("Gradient accumulation configured with %d steps.", steps)

    @property
    def steps(self) -> int:
        """Number of batches accumulated per optimizer step."""
        return self._steps

    @property
    def optimizer_step_idx(self) -> int:
        """Number of optimizer steps completed by this control."""
        return self._optimizer_step_idx

    def reset(self) -> None:
        """Reset optimizer-step progress for a new training run."""
        self._optimizer_step_idx = 0
        logger.debug("Gradient accumulation state reset.")

    def is_group_start(self, batch_idx: int) -> bool:
        """Return whether this batch begins an accumulation group."""
        return batch_idx % self._steps == 0

    def is_group_end(self, batch_idx: int, total_batches: int) -> bool:
        """Return whether this batch completes an accumulation group."""
        return (batch_idx + 1) % self._steps == 0 or batch_idx + 1 == total_batches

    def group_size(self, batch_idx: int, total_batches: int) -> int:
        """Return the actual size of the current accumulation group."""
        group_start = batch_idx - (batch_idx % self._steps)
        return min(self._steps, total_batches - group_start)

    def record_optimizer_step(self) -> int:
        """Record and return a completed optimizer-step index."""
        self._optimizer_step_idx += 1
        logger.debug(
            "Gradient accumulation completed optimizer step %d.",
            self._optimizer_step_idx,
        )
        return self._optimizer_step_idx

    def state_dict(self) -> dict[str, Any]:
        """Return resumable accumulation state."""
        return {
            "steps": self._steps,
            "optimizer_step_idx": self._optimizer_step_idx,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore accumulation state."""
        try:
            saved_steps = int(state_dict["steps"])
            optimizer_step_idx = int(state_dict["optimizer_step_idx"])
        except (KeyError, TypeError, ValueError) as error:
            logger.exception("Invalid gradient accumulation state.")
            msg = "Invalid GradientAccumulation checkpoint state."
            raise ValueError(msg) from error
        if saved_steps != self._steps:
            logger.error(
                "Gradient accumulation step mismatch: configured=%d, saved=%d",
                self._steps,
                saved_steps,
            )
            msg = "GradientAccumulation steps do not match checkpoint state."
            raise ValueError(msg)
        self._optimizer_step_idx = optimizer_step_idx
        logger.info(
            "Restored gradient accumulation at optimizer step %d.",
            self._optimizer_step_idx,
        )
