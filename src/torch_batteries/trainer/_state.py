"""Shared typing contract for private Battery workflow mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from pathlib import Path

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from torch_batteries.callbacks.base import Callback
    from torch_batteries.data import DataPack, ResolvedData
    from torch_batteries.data.handler import DataPackHandler
    from torch_batteries.data.types import DataStage
    from torch_batteries.events import EventHandler
    from torch_batteries.trainer.types import TrainResult
    from torch_batteries.utils.metrics import Metric, PhaseMetricManager

    from .core import Battery

    class BatteryStateMixin:
        """Describe state and cross-module operations supplied by Battery."""

        _callbacks: list[Callback]
        _data_pack: DataPack | None
        _data_pack_handler: DataPackHandler | None
        _device: torch.device
        _event_handler: EventHandler
        _last_completed_epoch: int
        _metric_manager: PhaseMetricManager
        _metrics: dict[str, Metric]
        _model: nn.Module
        _optimizer: torch.optim.Optimizer | None
        _optimizer_step_idx: int
        _resume_loaded: bool
        _stop_training: bool
        _train_results: TrainResult

        def _data_workflow(
            self,
            stage: DataStage,
            *,
            dataset_name: str | None = None,
        ) -> AbstractContextManager[ResolvedData]: ...

        @staticmethod
        def _validate_loader(dataloader: object, name: str) -> None: ...

        def _parse_step_result(
            self, result: Any, phase: str
        ) -> tuple[
            torch.Tensor,
            dict[str, float],
            torch.Tensor | None,
            torch.Tensor | None,
        ]: ...

        def _validate_train_inputs(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader | None,
        ) -> None: ...

        def load_checkpoint(self, path: str | Path) -> None: ...

        def _validate_epoch(
            self,
            dataloader: DataLoader,
            progress: Any,
            epoch: int,
        ) -> dict[str, float]: ...

else:

    class BatteryStateMixin:
        """Provide an empty shared runtime base for Battery workflow mixins."""

        __slots__ = ()


def as_battery(value: BatteryStateMixin) -> Battery:
    """Narrow an internal workflow mixin to the complete public Battery type.

    Args:
        value: Mixin instance owned by a complete Battery object.
    """
    return cast("Battery", value)
