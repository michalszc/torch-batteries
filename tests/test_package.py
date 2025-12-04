"""Tests for torch_batteries package imports and basic functionality."""

import torch
from torch import nn

import torch_batteries
from torch_batteries import events, trainer, utils
from torch_batteries.events import Event, EventHandler, charge
from torch_batteries.trainer import Battery
from torch_batteries.utils import batch, device, logging, progress


def test_package_import() -> None:
    """Test that the main package can be imported."""
    assert torch_batteries is not None
    assert hasattr(torch_batteries, "__version__")


def test_submodules_import() -> None:
    """Test that all submodules can be imported."""
    assert events is not None
    assert trainer is not None
    assert utils is not None


def test_core_classes_import() -> None:
    """Test that core classes can be imported."""
    assert Event is not None
    assert EventHandler is not None
    assert charge is not None
    assert Battery is not None


def test_utils_import() -> None:
    """Test that utility modules can be imported."""
    assert batch is not None
    assert device is not None
    assert logging is not None
    assert progress is not None


def test_basic_model_creation() -> None:
    """Test basic model and battery creation."""
    model = nn.Linear(10, 1)
    battery = Battery(model)

    assert battery.model is model
    assert isinstance(battery.device, torch.device)
    assert battery.optimizer is None
