"""Test for torch_batteries.callbacks.model_checkpoint.ModelCheckpoint module."""

from pathlib import Path
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
        assert checkpoint._stage == "val"  # noqa: SLF001
        assert checkpoint._metric == "accuracy"  # noqa: SLF001
        assert checkpoint._mode == "max"  # noqa: SLF001
        assert checkpoint._save_dir == "./checkpoints"  # noqa: SLF001
        assert checkpoint._save_path == "best_model.pth"  # noqa: SLF001
        assert checkpoint._save_top_k == 3  # noqa: SLF001
        assert checkpoint._verbose is True  # noqa: SLF001

    def test_invalid_stage(self) -> None:
        """Test ModelCheckpoint initialization with invalid stage."""
        with pytest.raises(ValueError, match="stage must be one of 'train' or 'val'"):
            ModelCheckpoint(stage="invalid", metric="accuracy")  # type: ignore[arg-type]

    def test_invalid_mode(self) -> None:
        """Test ModelCheckpoint initialization with invalid mode."""
        with pytest.raises(ValueError, match="mode must be one of 'min' or 'max'"):
            ModelCheckpoint(stage="val", metric="accuracy", mode="invalid")  # type: ignore[arg-type]

    def test_invalid_save_top_k(self) -> None:
        """At least one checkpoint must be retained."""
        with pytest.raises(ValueError, match="save_top_k"):
            ModelCheckpoint(stage="val", metric="loss", save_top_k=0)

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

    def test_callback_does_not_mutate_context_metrics(self, tmp_path: Path) -> None:
        """Adding filename fields does not leak into shared event metrics."""
        checkpoint = ModelCheckpoint(
            stage="val", metric="accuracy", save_dir=str(tmp_path)
        )
        metrics = {"accuracy": 0.8}
        context: EventContext = {
            "model": torch.nn.Linear(1, 1),
            "val_metrics": metrics,
            "epoch": 2,
        }

        checkpoint.run_on_validation_end(context)

        assert metrics == {"accuracy": 0.8}

    def test_creates_nested_directories_and_preserves_suffix(
        self, tmp_path: Path
    ) -> None:
        """Nested checkpoint templates and explicit suffixes are respected."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            save_dir=str(tmp_path / "missing"),
            save_path="nested/best.pt",
        )
        checkpoint.run_on_validation_end(
            {
                "model": torch.nn.Linear(1, 1),
                "val_metrics": {"accuracy": 0.8},
                "epoch": 1,
            }
        )

        assert checkpoint.best_model_path is not None
        path = Path(checkpoint.best_model_path)
        assert path.exists()
        assert path.suffix == ".pt"
        assert not path.name.endswith(".pt.pth")

    def test_adds_suffix_after_decimal_metric_value(self, tmp_path: Path) -> None:
        """Decimal metric formatting is not mistaken for a file suffix."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            save_dir=str(tmp_path),
            save_path="accuracy-{accuracy:.2f}",
        )
        checkpoint.run_on_validation_end(
            {
                "model": torch.nn.Linear(1, 1),
                "val_metrics": {"accuracy": 0.85},
                "epoch": 1,
            }
        )

        assert checkpoint.best_model_path is not None
        assert checkpoint.best_model_path.endswith(".pth")

    def test_static_top_k_template_gets_unique_epoch_paths(
        self, tmp_path: Path
    ) -> None:
        """Static top-k names cannot overwrite checkpoints from prior epochs."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            save_dir=str(tmp_path),
            save_path="best.pth",
            save_top_k=2,
        )
        model = torch.nn.Linear(1, 1)

        for epoch, accuracy in enumerate((0.8, 0.9)):
            checkpoint.run_on_validation_end(
                {
                    "model": model,
                    "val_metrics": {"accuracy": accuracy},
                    "epoch": epoch,
                }
            )

        paths = [Path(path) for path in checkpoint.best_k_models]
        assert len(paths) == 2
        assert len(set(paths)) == 2
        assert all(path.exists() and path.suffix == ".pth" for path in paths)

    def test_cleanup_tolerates_already_missing_file(self, tmp_path: Path) -> None:
        """Top-k cleanup remains safe when a checkpoint was removed externally."""
        checkpoint = ModelCheckpoint(
            stage="val", metric="accuracy", save_dir=str(tmp_path)
        )
        missing_path = tmp_path / "missing.pth"
        checkpoint._best_k_models[str(missing_path)] = 0.5  # noqa: SLF001

        checkpoint._delete_saved_model(str(missing_path))  # noqa: SLF001

        assert str(missing_path) not in checkpoint.best_k_models

    def test_min_mode_retains_two_lowest_checkpoints(self, tmp_path: Path) -> None:
        """Minimum-mode top-k retention evicts the highest loss checkpoint."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="loss",
            mode="min",
            save_dir=str(tmp_path),
            save_top_k=2,
        )
        model = torch.nn.Linear(1, 1)

        for epoch, loss in enumerate((0.5, 0.3, 0.4)):
            checkpoint.run_on_validation_end(
                {
                    "model": model,
                    "val_metrics": {"loss": loss},
                    "epoch": epoch,
                }
            )

        assert sorted(checkpoint.best_k_models.values()) == [0.3, 0.4]
        retained_paths = {Path(path) for path in checkpoint.best_k_models}
        assert all(path.exists() for path in retained_paths)
        checkpoint_files = set(tmp_path.glob("*.pth"))
        assert checkpoint_files == retained_paths
        assert not any("epoch=0" in path.name for path in checkpoint_files)
