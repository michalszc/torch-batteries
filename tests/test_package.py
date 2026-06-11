"""Tests for torch_batteries package imports and basic functionality."""

import builtins
import importlib
import sys
from typing import Any

import pytest
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


def test_plain_install_imports_without_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test package imports when the optional wandb dependency is unavailable."""
    real_import = builtins.__import__

    def block_wandb_import(name: str, *args: Any, **kwargs: Any) -> Any:
        level = kwargs.get(
            "level",
            args[-1] if args and isinstance(args[-1], int) else 0,
        )
        if level == 0 and (name == "wandb" or name.startswith("wandb.")):
            msg = "No module named 'wandb'"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monkeypatch.setattr(builtins, "__import__", block_wandb_import)

    tracking_module = importlib.import_module("torch_batteries.tracking")
    wandb_module = importlib.import_module("torch_batteries.tracking.wandb")

    importlib.reload(wandb_module)
    importlib.reload(tracking_module)
    reloaded_package = importlib.reload(torch_batteries)

    imported_battery = reloaded_package.Battery
    run = reloaded_package.Run
    wandb_tracker = tracking_module.WandbTracker

    assert imported_battery is Battery
    assert run is not None
    assert wandb_tracker is not None

    with pytest.raises(ImportError, match=r"pip install torch-batteries\[wandb\]"):
        wandb_tracker(project="test-project")
