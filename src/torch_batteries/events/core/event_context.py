"""Core events and decorators for torch-batteries."""

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn
from typing_extensions import TypedDict

from .optimization_step import OptimizationStep

if TYPE_CHECKING:
    import torch_batteries


class EventContext(TypedDict, total=False):
    """Context dictionary passed to event handlers.

    Different events populate different keys. All keys are optional so the same
    type can describe training, validation, testing, and prediction events.

    Common keys:

    - `battery`: The `Battery` instance managing the workflow.
    - `model`: The model/module being trained, validated, tested, or used for
      prediction.
    - `optimizer`: The optimizer when available.
    - `batch`: Current batch data, usually a tuple or list of tensors.
    - `batch_idx`: Current batch index within the active phase.
    - `epoch`: One-based public epoch number. Training and validation workflows
      expose `1, 2, 3, ...`; single-pass test and prediction workflows expose
      `1`.
    - `device`: Device selected by Battery.
    - `phase`: Active workflow phase: `train`, `validation`, `test`, or
      `predict`.
    - `dataset_name`: Name of the active implicit DataPack dataset.

    Optimization keys:

    - `total_batches`: Number of batches in the active training loader.
    - `optimization_plan`: Zeroing, loss-scaling, and optimizer-boundary plan.
    - `loss_tensor`: Original scalar loss returned by the training step.
    - `backward_loss`: Loss tensor that will be passed to backward. A
      `BEFORE_BACKWARD` handler may replace it.
    - `optimizer_step`: Whether the current batch performs a real optimizer
      step.
    - `optimizer_step_idx`: Number of successfully completed optimizer steps.

    Loss keys:

    - `train_loss`: Current training loss for training step events.
    - `val_loss`: Current validation loss for validation step events.
    - `test_loss`: Current test loss for test events.
    - `loss`: Deprecated compatibility alias for the phase-specific loss key.

    Metric keys:

    - `train_metrics`: Current training batch or epoch metrics.
    - `val_metrics`: Current validation batch or epoch metrics.
    - `test_metrics`: Current test batch or final metrics.

    History keys:

    - `history_train_loss`: Training loss history for completed epochs.
    - `history_val_loss`: Validation loss history for completed epochs.
    - `history_train_metrics`: Training metric history for completed epochs.
    - `history_val_metrics`: Validation metric history for completed epochs.

    Prediction keys:

    - `predictions`: Model predictions from a prediction step or prediction run.
    """

    battery: "torch_batteries.Battery"
    model: nn.Module
    optimizer: torch.optim.Optimizer | None
    device: torch.device
    phase: Literal["train", "validation", "test", "predict"]
    dataset_name: str
    batch: Any
    batch_idx: int
    total_batches: int
    epoch: int
    optimization_plan: OptimizationStep
    loss_tensor: torch.Tensor
    backward_loss: torch.Tensor
    loss: float
    train_loss: float
    val_loss: float
    test_loss: float
    predictions: Any
    prediction_batches: int
    optimizer_step: bool
    optimizer_step_idx: int
    resumed: bool
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    history_train_loss: list[float]
    history_val_loss: list[float]
    history_train_metrics: dict[str, list[float]]
    history_val_metrics: dict[str, list[float]]
