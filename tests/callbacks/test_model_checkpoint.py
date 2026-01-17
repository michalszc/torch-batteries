"""Test for torch_batteries.callbacks.model_checkpoint.ModelCheckpoint module."""

from typing import TYPE_CHECKING

import pytest
import torch

from torch_batteries.callbacks.model_checkpoint import ModelCheckpoint

if TYPE_CHECKING:
    from torch_batteries.events import EventContext


class TestModelCheckpoint:
    """Test cases for ModelCheckpoint callback."""

    def test_initialization(self) -> None:
        """Test ModelCheckpoint initialization with valid parameters."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            mode="max",
            save_dir="./checkpoints",
            save_path="best_model.pth",
            save_top_k=3,
            verbose=True,
        )
        assert checkpoint.stage == "val"
        assert checkpoint.metric == "accuracy"
        assert checkpoint.mode == "max"
        assert checkpoint.save_dir == "./checkpoints"
        assert checkpoint.save_path == "best_model.pth"
        assert checkpoint.save_top_k == 3
        assert checkpoint.verbose is True

    def test_invalid_stage(self) -> None:
        """Test ModelCheckpoint initialization with invalid stage."""
        with pytest.raises(ValueError, match="stage must be one of 'train' or 'val'"):
            ModelCheckpoint(stage="invalid", metric="accuracy")  # type: ignore[arg-type]

    def test_invalid_mode(self) -> None:
        """Test ModelCheckpoint initialization with invalid mode."""
        with pytest.raises(ValueError, match="mode must be one of 'min' or 'max'"):
            ModelCheckpoint(stage="val", metric="accuracy", mode="invalid")

    def test_run_on_validation_end(self, tmp_path: object) -> None:
        """Test run_on_validation_end method."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            mode="max",
            save_dir=str(tmp_path),
            save_top_k=1,
            verbose=True,
        )
        model = torch.nn.Linear(1, 1)
        context: EventContext = {
            "model": model,
            "val_metrics": {"accuracy": 0.85},
            "epoch": 1,
        }

        checkpoint.run_on_validation_end(context)
        assert checkpoint.best_model_path is not None
        assert torch.load(checkpoint.best_model_path) is not None

    def test_run_on_train_epoch_end(self, tmp_path: object) -> None:
        """Test run_on_test_end method."""
        checkpoint = ModelCheckpoint(
            stage="train",
            metric="accuracy",
            mode="max",
            save_dir=str(tmp_path),
            save_top_k=1,
            verbose=True,
        )
        model = torch.nn.Linear(1, 1)
        context: EventContext = {
            "model": model,
            "train_metrics": {"accuracy": 0.9},
            "epoch": 1,
        }

        checkpoint.run_on_train_epoch_end(context)
        assert checkpoint.best_model_path is not None
        assert torch.load(checkpoint.best_model_path) is not None

    def test_save_best_model(self, tmp_path: object) -> None:
        """Test saving the best model."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            mode="max",
            save_dir=str(tmp_path),
            save_top_k=1,
            verbose=True,
        )
        model = torch.nn.Linear(1, 1)

        context: EventContext = {
            "model": model,
            "val_metrics": {"accuracy": 0.8},
            "epoch": 1,
        }
        checkpoint.run_on_validation_end(context)
        assert checkpoint.best_model_path is not None
        assert torch.load(checkpoint.best_model_path) is not None

        context = {
            "model": model,
            "val_metrics": {"accuracy": 0.85},
            "epoch": 2,
        }
        checkpoint.run_on_validation_end(context)
        assert checkpoint.best_model_path is not None
        assert torch.load(checkpoint.best_model_path) is not None

    def test_save_top_k_model(self, tmp_path: object) -> None:
        """Test saving top K models."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            mode="max",
            save_dir=str(tmp_path),
            save_top_k=2,
            verbose=True,
        )
        model = torch.nn.Linear(1, 1)

        context: EventContext = {
            "model": model,
            "val_metrics": {"accuracy": 0.75},
            "epoch": 1,
        }

        checkpoint.run_on_validation_end(context)
        assert len(checkpoint.best_k_models) == 1

        context = {
            "model": model,
            "val_metrics": {"accuracy": 0.85},
            "epoch": 2,
        }
        checkpoint.run_on_validation_end(context)
        assert len(checkpoint.best_k_models) == 2

        context = {
            "model": model,
            "val_metrics": {"accuracy": 0.65},
            "epoch": 3,
        }
        checkpoint.run_on_validation_end(context)
        assert len(checkpoint.best_k_models) == 2

        context = {
            "model": model,
            "val_metrics": {"accuracy": 0.9},
            "epoch": 4,
        }

        checkpoint.run_on_validation_end(context)
        assert len(checkpoint.best_k_models) == 2

        assert all(score >= 0.85 for score in checkpoint.best_k_models.values())
