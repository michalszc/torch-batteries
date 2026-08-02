"""Mixed-precision training control."""

from contextlib import AbstractContextManager
from typing import Any, Literal

import torch

from torch_batteries.callbacks.base import Callback
from torch_batteries.events import Event, EventContext, charge
from torch_batteries.utils.logging import get_logger

logger = get_logger("MixedPrecision")

Precision = Literal["32-true", "16-mixed", "bf16-mixed", "amp"]
_VALID_PRECISIONS = {"32-true", "16-mixed", "bf16-mixed", "amp"}


class MixedPrecision(Callback):
    """Apply full or mixed precision across all Battery workflow phases."""

    __slots__ = ("_device", "_effective_precision", "_precision", "_scaler")

    def __init__(self, precision: Precision = "amp") -> None:
        if precision not in _VALID_PRECISIONS:
            logger.error("Unsupported precision mode: %s", precision)
            msg = (
                "precision must be one of '32-true', '16-mixed', "
                "'bf16-mixed', or 'amp'."
            )
            raise ValueError(msg)
        self._precision = precision
        self._effective_precision: Precision = precision
        self._device = torch.device("cpu")
        self._scaler = torch.amp.GradScaler("cpu", enabled=False)
        logger.debug("Mixed precision callback created with mode %s.", precision)

    @property
    def precision(self) -> Precision:
        """Requested precision mode."""
        return self._precision

    @property
    def effective_precision(self) -> Precision:
        """Device-resolved precision mode."""
        return self._effective_precision

    @property
    def scaler(self) -> torch.amp.GradScaler:
        """Gradient scaler used by fp16 training."""
        return self._scaler

    def configure(self, device: torch.device) -> None:
        """Resolve the requested precision for a concrete device."""
        self._device = device
        if self._precision == "amp":
            self._effective_precision = (
                "bf16-mixed" if device.type == "cpu" else "16-mixed"
            )
        else:
            self._effective_precision = self._precision

        if device.type not in {"cpu", "cuda", "mps"}:
            logger.error(
                "Mixed precision is unsupported on device type %s.", device.type
            )
            msg = f"MixedPrecision does not support device type '{device.type}'."
            raise ValueError(msg)

        self._scaler = torch.amp.GradScaler(
            device.type,
            enabled=self._effective_precision == "16-mixed",
        )
        logger.info(
            "Precision configured: requested=%s, effective=%s, device=%s",
            self._precision,
            self._effective_precision,
            device,
        )

    def autocast(self) -> AbstractContextManager[None]:
        """Create an autocast context for the configured precision."""
        if self._effective_precision == "32-true":
            return torch.autocast(self._device.type, enabled=False)
        dtype = (
            torch.float16 if self._effective_precision == "16-mixed" else torch.bfloat16
        )
        return torch.autocast(self._device.type, dtype=dtype)

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate a normalized loss with optional gradient scaling."""
        self._scaler.scale(loss).backward()
        logger.debug(
            "Mixed precision backward completed with scaler enabled=%s.",
            self._scaler.is_enabled(),
        )

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Apply an optimizer step and update the gradient scaler."""
        self._scaler.step(optimizer)
        self._scaler.update()
        logger.debug("Mixed precision optimizer step completed.")

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale optimizer gradients before operations such as clipping."""
        if self._scaler.is_enabled():
            self._scaler.unscale_(optimizer)
            logger.debug("Mixed precision gradients unscaled.")

    @charge(Event.SETUP)
    def on_setup(self, context: EventContext) -> None:
        """Resolve precision using Battery's selected device."""
        self.configure(context["device"])

    @charge(Event.STEP_EXECUTION_CONTEXT)
    def step_execution_context(
        self, context: EventContext
    ) -> AbstractContextManager[None]:
        """Provide autocast for train, validation, test, and prediction steps."""
        logger.debug(
            "Providing mixed-precision context for phase %s.", context["phase"]
        )
        return self.autocast()

    @charge(Event.BACKWARD)
    def run_backward(self, context: EventContext) -> None:
        """Execute scaled or ordinary backward through the event contract."""
        self.backward(context["backward_loss"])

    @charge(Event.BEFORE_GRADIENT_CLIP)
    def prepare_gradients(self, context: EventContext) -> None:
        """Unscale gradients before optional clipping."""
        optimizer = context["optimizer"]
        if optimizer is None:
            logger.error("Mixed precision requires an optimizer before clipping.")
            msg = "MixedPrecision requires an optimizer for gradient preparation."
            raise ValueError(msg)
        self.unscale_(optimizer)

    @charge(Event.OPTIMIZER_STEP)
    def run_optimizer_step(self, context: EventContext) -> None:
        """Execute the scaler-aware optimizer step."""
        optimizer = context["optimizer"]
        if optimizer is None:
            logger.error("Mixed precision requires an optimizer for optimizer step.")
            msg = "MixedPrecision requires an optimizer for optimizer step."
            raise ValueError(msg)
        self.optimizer_step(optimizer)

    def state_dict(self) -> dict[str, Any]:
        """Return precision and scaler state."""
        return {
            "precision": self._precision,
            "effective_precision": self._effective_precision,
            "scaler": self._scaler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scaler state for a compatible precision configuration."""
        try:
            saved_precision = state_dict["precision"]
            saved_effective = state_dict["effective_precision"]
            scaler_state = state_dict["scaler"]
        except KeyError as error:
            logger.exception("Invalid mixed precision state.")
            msg = "Invalid MixedPrecision checkpoint state."
            raise ValueError(msg) from error
        if (
            saved_precision != self._precision
            or saved_effective != self._effective_precision
        ):
            logger.error(
                "Mixed precision state mismatch: configured=%s/%s, saved=%s/%s",
                self._precision,
                self._effective_precision,
                saved_precision,
                saved_effective,
            )
            msg = "MixedPrecision configuration does not match checkpoint state."
            raise ValueError(msg)
        if not isinstance(scaler_state, dict):
            logger.error("Mixed precision scaler state is not a dictionary.")
            msg = "Invalid MixedPrecision scaler checkpoint state."
            raise TypeError(msg)
        self._scaler.load_state_dict(scaler_state)
        logger.info("Mixed precision scaler state restored.")
