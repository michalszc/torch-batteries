"""Gradient accumulation training control."""

from typing import Any

from torch_batteries.callbacks.base import Callback
from torch_batteries.events import Event, EventContext, OptimizationStep, charge
from torch_batteries.utils.logging import get_logger

logger = get_logger("callbacks.gradient_accumulation")


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

    @charge(Event.BEFORE_TRAIN)
    def on_train_start(self, context: EventContext) -> None:
        """Reset progress when training is not resuming from a checkpoint."""
        if not context.get("resumed", False):
            self.reset()

    @charge(Event.CONFIGURE_TRAIN_STEP)
    def configure_train_step(self, context: EventContext) -> OptimizationStep:
        """Return zeroing, scaling, and step decisions for the current batch."""
        batch_idx = context["batch_idx"]
        total_batches = context["total_batches"]
        plan = OptimizationStep(
            zero_grad=self.is_group_start(batch_idx),
            optimizer_step=self.is_group_end(batch_idx, total_batches),
            loss_divisor=self.group_size(batch_idx, total_batches),
        )
        logger.debug(
            "Gradient accumulation plan: batch=%d, zero_grad=%s, "
            "optimizer_step=%s, loss_divisor=%d",
            batch_idx,
            plan.zero_grad,
            plan.optimizer_step,
            plan.loss_divisor,
        )
        return plan

    @charge(Event.AFTER_OPTIMIZER_STEP)
    def on_optimizer_step_end(self, context: EventContext) -> None:
        """Synchronize callback state with Battery's completed-step counter."""
        self._optimizer_step_idx = context["optimizer_step_idx"]
        logger.debug(
            "Gradient accumulation synchronized at optimizer step %d.",
            self._optimizer_step_idx,
        )

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
