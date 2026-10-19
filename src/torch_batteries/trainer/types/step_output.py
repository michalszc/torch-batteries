"""Explicit output from model training and evaluation steps."""

from dataclasses import dataclass, field
from typing import Self

import torch

type StructuredTensors = torch.Tensor | dict[str, torch.Tensor]


@dataclass(slots=True)
class StepOutput:
    """Explicit output from a training, validation, or test step.

    Outputs can be combined with ``+``, for example to build a multi-head step
    from one output per head. Losses are summed, while predictions, targets, and
    metrics are merged by key. ``sum(outputs)`` is supported as well.

    Args:
        loss: Scalar tensor used for reporting and, during training, backward.
        predictions: Outputs produced by the same forward pass as ``loss``.
        targets: Ground-truth tensors corresponding to ``predictions``.
        metrics: Named scalar tensors or numeric values calculated by the step.

    Raises:
        ValueError: On ``+`` when both sides share a predictions, targets, or
            metrics key.
        TypeError: On ``+`` when both sides set predictions or targets and they
            are not both dictionaries.
    """

    loss: torch.Tensor
    predictions: StructuredTensors | None = None
    targets: StructuredTensors | None = None
    metrics: dict[str, float | torch.Tensor] = field(default_factory=dict)

    def __add__(self, other: object) -> Self:
        if not isinstance(other, StepOutput):
            return NotImplemented
        return type(self)(
            loss=self.loss + other.loss,
            predictions=self._merge_structured(
                self.predictions, other.predictions, "predictions"
            ),
            targets=self._merge_structured(self.targets, other.targets, "targets"),
            metrics=self._merge_keyed(self.metrics, other.metrics, "metrics"),
        )

    def __radd__(self, other: object) -> Self:
        # Lets ``sum(outputs)`` start from its default integer zero.
        if isinstance(other, int) and other == 0:
            return self
        return NotImplemented

    @staticmethod
    def _merge_keyed[T](
        left: dict[str, T], right: dict[str, T], field_name: str
    ) -> dict[str, T]:
        """Merge two mappings, rejecting keys present in both."""
        duplicates = left.keys() & right.keys()
        if duplicates:
            msg = (
                f"Cannot add StepOutput values with duplicate {field_name} keys: "
                f"{sorted(duplicates)}."
            )
            raise ValueError(msg)
        return {**left, **right}

    @staticmethod
    def _merge_structured(
        left: StructuredTensors | None,
        right: StructuredTensors | None,
        field_name: str,
    ) -> StructuredTensors | None:
        """Combine structured tensors when at most one side is set or both are keyed."""
        if left is None:
            return right
        if right is None:
            return left
        if isinstance(left, dict) and isinstance(right, dict):
            return StepOutput._merge_keyed(left, right, field_name)
        msg = (
            f"Cannot add StepOutput values whose {field_name} are both set unless "
            "both are dictionaries with distinct keys."
        )
        raise TypeError(msg)
