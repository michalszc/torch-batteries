"""Compatibility coverage for public imports preserved by package facades."""

import torch_batteries
from torch_batteries import data, events, tracking, trainer
from torch_batteries.data import types as data_types
from torch_batteries.events import core as event_core
from torch_batteries.tracking import wandb
from torch_batteries.trainer import types as trainer_types
from torch_batteries.utils import logging, metrics
from torch_batteries.utils.progress import types as progress_types


def test_root_and_component_exports_retain_identity() -> None:
    """Package-root imports continue to resolve through component facades."""
    assert torch_batteries.Battery is trainer.Battery
    assert torch_batteries.DataPack is data.DataPack
    assert torch_batteries.DataPackHandler is data.DataPackHandler
    assert torch_batteries.Event is events.Event
    assert torch_batteries.EventContext is events.EventContext
    assert torch_batteries.OptimizationStep is events.OptimizationStep
    assert torch_batteries.charge is events.charge


def test_historical_type_facades_retain_exports() -> None:
    """Existing type-module imports remain valid after conversion to packages."""
    assert data_types.DataContext is data.DataContext
    assert data_types.DataLoaderBundle is data.DataLoaderBundle
    assert data_types.DataLoaderConfig is data.DataLoaderConfig
    assert data_types.DatasetBundle is data.DatasetBundle
    assert data_types.ResolvedData is data.ResolvedData
    assert event_core.Event is events.Event
    assert event_core.EventContext is events.EventContext
    assert event_core.OptimizationStep is events.OptimizationStep
    assert event_core.charge is events.charge
    assert trainer_types.PredictResult is trainer.PredictResult
    assert trainer_types.FitResult is trainer.FitResult
    assert trainer_types.StepOutput is trainer.StepOutput
    assert trainer_types.TestResult is trainer.TestResult
    assert trainer_types.TrainResult is trainer.TrainResult
    assert trainer_types.ValidationResult is trainer.ValidationResult


def test_utility_and_integration_facades_retain_exports() -> None:
    """Utility and optional-integration import paths remain stable."""
    assert metrics.CollectedMetric is torch_batteries.CollectedMetric
    assert metrics.StatefulMetric is torch_batteries.StatefulMetric
    assert callable(metrics.calculate_metrics)
    assert callable(logging.get_logger)
    assert progress_types.Phase is not None
    assert progress_types.ProgressMetrics is not None
    assert wandb.WandbTracker is tracking.WandbTracker
