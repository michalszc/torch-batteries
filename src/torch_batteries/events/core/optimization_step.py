"""Core events and decorators for torch-batteries."""

from dataclasses import dataclass

from torch_batteries.utils.logging import get_logger

logger = get_logger("events.core")


@dataclass(frozen=True, slots=True)
class OptimizationStep:
    """Describe the gradient operations required for one training batch.

    Returned by a model or callback handling
    :attr:`Event.CONFIGURE_TRAIN_STEP`. With no handler, Battery uses these
    defaults to zero gradients, backpropagate the full loss, and perform one
    optimizer step per batch.

    Args:
        zero_grad: Whether gradients are cleared before the model step.
        optimizer_step: Whether this batch completes an optimizer group.
        loss_divisor: Positive divisor applied to the loss before backward.
    """

    zero_grad: bool = True
    optimizer_step: bool = True
    loss_divisor: int = 1

    def __post_init__(self) -> None:
        """Validate loss normalization before the plan reaches the trainer."""
        if isinstance(self.loss_divisor, bool) or not isinstance(
            self.loss_divisor, int
        ):
            logger.error(
                "Invalid optimization-step loss divisor: %r", self.loss_divisor
            )
            msg = "OptimizationStep loss_divisor must be an integer."
            raise TypeError(msg)
        if self.loss_divisor < 1:
            logger.error(
                "Invalid optimization-step loss divisor: %r", self.loss_divisor
            )
            msg = "OptimizationStep loss_divisor must be a positive integer."
            raise ValueError(msg)
