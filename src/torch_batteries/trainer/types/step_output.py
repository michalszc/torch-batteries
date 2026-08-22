"""Explicit output from model training and evaluation steps."""

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class StepOutput:
    """Explicit output from a training, validation, or test step.

    Args:
        loss: Scalar tensor used for reporting and, during training, backward.
        predictions: Outputs produced by the same forward pass as ``loss``.
        targets: Ground-truth tensors corresponding to ``predictions``.
        metrics: Named scalar tensors or numeric values calculated by the step.
    """

    loss: torch.Tensor
    predictions: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    metrics: dict[str, float | torch.Tensor] = field(default_factory=dict)
