"""Terminate workflows when selected losses or metrics are not finite."""

import math
from typing import Any

import torch

from torch_batteries.events import Event, EventContext, charge
from torch_batteries.utils.logging import get_logger

from .base import Callback

logger = get_logger("callbacks.terminate_on_non_finite")


class TerminateOnNonFinite(Callback):
    """Raise when a workflow produces a selected NaN or infinite value.

    Args:
        check_loss: Check training loss before backward and evaluation losses at
            step completion.
        check_metrics: Check named batch metrics and final phase metrics.

    Raises:
        TypeError: If either option is not a boolean.
        ValueError: If both checks are disabled.
        FloatingPointError: When a selected value is NaN or infinite.
    """

    __slots__ = ("_check_loss", "_check_metrics")

    def __init__(self, *, check_loss: bool = True, check_metrics: bool = True) -> None:
        self._validate_configuration(check_loss, check_metrics)
        self._check_loss = check_loss
        self._check_metrics = check_metrics

    @staticmethod
    def _validate_configuration(check_loss: object, check_metrics: object) -> None:
        """Validate non-finite options from any configuration source."""
        if not isinstance(check_loss, bool) or not isinstance(check_metrics, bool):
            msg = "check_loss and check_metrics must be booleans."
            raise TypeError(msg)
        if not check_loss and not check_metrics:
            msg = "At least one of check_loss or check_metrics must be enabled."
            raise ValueError(msg)

    def state_dict(self) -> dict[str, bool]:
        """Return the fixed non-finite check configuration."""
        return {
            "check_loss": self._check_loss,
            "check_metrics": self._check_metrics,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate checkpoint configuration without changing this callback.

        Args:
            state_dict: Configuration stored by :meth:`state_dict`.
        """
        self._validate_checkpoint_state(state_dict)

    def _validate_checkpoint_state(self, state_dict: dict[str, Any]) -> None:
        """Validate fixed non-finite checks without changing state."""
        self._validate_configuration(
            state_dict.get("check_loss"), state_dict.get("check_metrics")
        )
        if state_dict != self.state_dict():
            logger.error("TerminateOnNonFinite checkpoint configuration mismatch.")
            msg = "TerminateOnNonFinite checkpoint configuration does not match."
            raise ValueError(msg)

    @charge(Event.BEFORE_BACKWARD)
    def on_before_backward(self, context: EventContext) -> None:
        """Check training loss before backward or optimizer execution.

        Args:
            context: Training optimization context containing ``loss_tensor``.
        """
        if self._check_loss:
            self._check_value("loss", context["loss_tensor"], "train", context)

    @charge(Event.AFTER_TRAIN_STEP)
    @charge(Event.AFTER_VALIDATION_STEP)
    @charge(Event.AFTER_TEST_STEP)
    def on_step_end(self, context: EventContext) -> None:
        """Check evaluation loss and named metrics after a phase step.

        Args:
            context: Completed train, validation, or test step context.
        """
        phase, loss, metrics = self._phase_values(context)
        if self._check_loss and phase != "train":
            self._check_value("loss", loss, phase, context)
        self._check_metric_values(metrics, phase, context)

    @charge(Event.AFTER_TRAIN_EPOCH)
    @charge(Event.AFTER_VALIDATION_EPOCH)
    @charge(Event.AFTER_TEST_EPOCH)
    def on_phase_end(self, context: EventContext) -> None:
        """Check final aggregated loss and stateful or collected metrics.

        Args:
            context: Completed train, validation, or test phase context.
        """
        phase, loss, metrics = self._phase_values(context)
        if self._check_loss:
            self._check_value(
                "loss",
                metrics.get("loss", loss),
                phase,
                context,
            )
        self._check_metric_values(metrics, phase, context)

    def _check_metric_values(
        self,
        metrics: dict[str, float],
        phase: str,
        context: EventContext,
    ) -> None:
        """Check enabled entries in one phase metric mapping."""
        for name, value in metrics.items():
            if name == "loss":
                if self._check_loss:
                    self._check_value(name, value, phase, context)
            elif self._check_metrics:
                self._check_value(name, value, phase, context)

    @staticmethod
    def _phase_values(
        context: EventContext,
    ) -> tuple[str, float | None, dict[str, float]]:
        """Infer phase, loss, and metrics from a dedicated lifecycle context."""
        if "train_metrics" in context:
            return "train", context.get("train_loss"), context["train_metrics"]
        if "val_metrics" in context:
            return "validation", context.get("val_loss"), context["val_metrics"]
        return "test", context.get("test_loss"), context["test_metrics"]

    @staticmethod
    def _check_value(
        name: str,
        value: Any,
        phase: str,
        context: EventContext,
    ) -> None:
        """Raise a contextual error for one non-finite scalar or tensor."""
        if isinstance(value, torch.Tensor):
            finite = bool(torch.isfinite(value).all().item())
        else:
            finite = math.isfinite(float(value))
        if finite:
            return

        details = [f"name={name!r}", f"value={value!r}", f"phase={phase!r}"]
        if "epoch" in context:
            details.append(f"epoch={context['epoch']}")
        if "batch_idx" in context:
            details.append(f"batch={context['batch_idx']}")
        description = ", ".join(details)
        logger.error("Non-finite workflow value detected: %s", description)
        msg = f"Non-finite workflow value detected: {description}."
        raise FloatingPointError(msg)
