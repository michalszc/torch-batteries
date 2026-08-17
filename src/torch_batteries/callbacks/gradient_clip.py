"""Gradient clipping training control."""

from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import nn

from torch_batteries.callbacks.base import Callback
from torch_batteries.events import Event, EventContext, charge
from torch_batteries.utils.logging import get_logger

logger = get_logger("callbacks.gradient_clip")

ClipAlgorithm = Literal["norm", "value"]


class GradientClip(Callback):
    """Clip gradients immediately before optimizer steps.

    ``norm`` scales all gradients proportionally when their combined norm exceeds
    ``value``. ``value`` clamps each gradient element independently.
    """

    __slots__ = ("_algorithm", "_value")

    def __init__(self, value: float, algorithm: ClipAlgorithm = "norm") -> None:
        if value < 0:
            logger.error("Gradient clip value must not be negative: %s", value)
            msg = "GradientClip value must be greater than or equal to zero."
            raise ValueError(msg)
        if algorithm not in {"norm", "value"}:
            logger.error("Unsupported gradient clipping algorithm: %s", algorithm)
            msg = "GradientClip algorithm must be 'norm' or 'value'."
            raise ValueError(msg)
        self._value = float(value)
        self._algorithm = algorithm
        logger.info(
            "Gradient clipping configured: algorithm=%s, value=%s",
            algorithm,
            self._value,
        )

    @property
    def value(self) -> float:
        """Configured clipping threshold."""
        return self._value

    @property
    def algorithm(self) -> ClipAlgorithm:
        """Configured clipping algorithm."""
        return self._algorithm

    def apply(self, parameters: Iterable[nn.Parameter]) -> float | None:
        """Clip gradients and return the pre-clip norm when available."""
        parameter_list = [
            parameter for parameter in parameters if parameter.grad is not None
        ]
        if self._algorithm == "norm":
            norm = torch.nn.utils.clip_grad_norm_(parameter_list, self._value)
            norm_value = float(norm)
            logger.debug(
                "Gradient norm clipping applied: pre_clip_norm=%s, max_norm=%s",
                norm_value,
                self._value,
            )
            return norm_value

        torch.nn.utils.clip_grad_value_(parameter_list, self._value)
        logger.debug("Gradient value clipping applied: clip_value=%s", self._value)
        return None

    @charge(Event.GRADIENT_CLIP)
    def run_gradient_clip(self, context: EventContext) -> None:
        """Clip gradients exposed by the optimization event context."""
        self.apply(context["model"].parameters())

    def state_dict(self) -> dict[str, Any]:
        """Return clipping configuration for strict checkpoint validation."""
        return {"value": self._value, "algorithm": self._algorithm}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate restored clipping configuration."""
        if (
            state_dict.get("value") != self._value
            or state_dict.get("algorithm") != self._algorithm
        ):
            logger.error(
                "Gradient clipping state mismatch: configured=%s/%s, saved=%s/%s",
                self._algorithm,
                self._value,
                state_dict.get("algorithm"),
                state_dict.get("value"),
            )
            msg = "GradientClip configuration does not match checkpoint state."
            raise ValueError(msg)
        logger.debug("Gradient clipping checkpoint configuration validated.")
