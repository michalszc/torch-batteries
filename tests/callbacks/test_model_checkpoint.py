"""Test for torch_batteries.callbacks.model_checkpoint.ModelCheckpoint module."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from torch_batteries.callbacks.model_checkpoint import ModelCheckpoint

if TYPE_CHECKING:
    from torch_batteries.events import EventContext


def valid_checkpoint_state() -> dict[str, object]:
    """Return a minimal valid callback state for corruption tests."""
    return {
        "best_k_models": {"first.pth": 0.8, "second.pth": 0.9},
        "best_model_path": "second.pth",
        "kth_best_model_path": "first.pth",
        "best_score": 0.9,
        "kth_best_score": 0.8,
        "save_weights_only": False,
    }


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
        )
        assert checkpoint._stage == "val"  # noqa: SLF001
        assert checkpoint._metric == "accuracy"  # noqa: SLF001
        assert checkpoint._mode == "max"  # noqa: SLF001
        assert checkpoint._save_dir == "./checkpoints"  # noqa: SLF001
        assert checkpoint._save_path == "best_model.pth"  # noqa: SLF001
        assert checkpoint._save_top_k == 3  # noqa: SLF001

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

    def test_checkpoint_filename_uses_event_epoch(self, tmp_path: Path) -> None:
        """The public one-based event epoch is preserved in the filename."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            save_dir=str(tmp_path),
            save_path="{epoch}-{accuracy:.2f}",
            save_weights_only=True,
        )

        checkpoint.run_on_validation_end(
            {
                "model": torch.nn.Linear(1, 1),
                "val_metrics": {"accuracy": 0.97},
                "epoch": 1,
            }
        )

        assert checkpoint.best_model_path is not None
        assert Path(checkpoint.best_model_path).name == "epoch=1-accuracy=0.97.pth"

    def test_save_best_model(self, tmp_path: object) -> None:
        """Test saving the best model."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            mode="max",
            save_dir=str(tmp_path),
            save_top_k=1,
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

    def test_top_one_skips_worse_checkpoint_without_cleanup_warning(
        self, tmp_path: Path
    ) -> None:
        """A non-best candidate is neither saved nor passed to cleanup."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            mode="max",
            save_dir=str(tmp_path),
            save_path="{epoch}-{accuracy:.2f}",
            save_top_k=1,
            save_weights_only=True,
        )
        model = torch.nn.Linear(1, 1)

        with patch(
            "torch_batteries.callbacks.model_checkpoint.logger.warning"
        ) as mock_warning:
            for epoch, accuracy in enumerate((0.9722, 0.9823, 0.9797), start=1):
                checkpoint.run_on_validation_end(
                    {
                        "model": model,
                        "val_metrics": {"accuracy": accuracy},
                        "epoch": epoch,
                    }
                )

        assert mock_warning.call_count == 0
        assert list(checkpoint.best_k_models.values()) == [0.9823]
        retained_path = Path(next(iter(checkpoint.best_k_models)))
        assert retained_path.exists()
        assert retained_path.name == "epoch=2-accuracy=0.98.pth"
        assert set(tmp_path.glob("*.pth")) == {retained_path}

    def test_replacement_is_saved_before_displaced_checkpoint_is_deleted(
        self, tmp_path: Path
    ) -> None:
        """Cleanup starts only after the accepted replacement exists."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            mode="max",
            save_dir=str(tmp_path),
            save_path="{epoch}-{accuracy:.2f}",
            save_top_k=1,
            save_weights_only=True,
        )
        model = torch.nn.Linear(1, 1)
        checkpoint.run_on_validation_end(
            {
                "model": model,
                "val_metrics": {"accuracy": 0.8},
                "epoch": 1,
            }
        )
        displaced_path = Path(next(iter(checkpoint.best_k_models)))

        original_delete = checkpoint._delete_checkpoint_file  # noqa: SLF001

        def assert_replacement_exists(filepath: str) -> None:
            assert filepath == str(displaced_path)
            assert len(list(tmp_path.glob("*.pth"))) == 2
            original_delete(filepath)

        with patch.object(
            checkpoint,
            "_delete_checkpoint_file",
            side_effect=assert_replacement_exists,
        ):
            checkpoint.run_on_validation_end(
                {
                    "model": model,
                    "val_metrics": {"accuracy": 0.9},
                    "epoch": 2,
                }
            )

        retained_path = Path(next(iter(checkpoint.best_k_models)))
        assert retained_path.exists()
        assert not displaced_path.exists()

    def test_failed_replacement_restores_checkpoint_ranking(
        self, tmp_path: Path
    ) -> None:
        """A failed write leaves the previously retained checkpoint authoritative."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            mode="max",
            save_dir=str(tmp_path),
            save_path="{epoch}-{accuracy:.2f}",
            save_top_k=1,
            save_weights_only=True,
        )
        model = torch.nn.Linear(1, 1)
        checkpoint.run_on_validation_end(
            {
                "model": model,
                "val_metrics": {"accuracy": 0.8},
                "epoch": 1,
            }
        )
        retained_path = Path(next(iter(checkpoint.best_k_models)))

        with (
            patch(
                "torch_batteries.callbacks.model_checkpoint.torch.save",
                side_effect=OSError("disk unavailable"),
            ),
            pytest.raises(OSError, match="disk unavailable"),
        ):
            checkpoint.run_on_validation_end(
                {
                    "model": model,
                    "val_metrics": {"accuracy": 0.9},
                    "epoch": 2,
                }
            )

        assert checkpoint.best_model_path == str(retained_path)
        assert checkpoint.best_score == 0.8
        assert checkpoint.best_k_models == {str(retained_path): 0.8}
        assert retained_path.exists()

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

        with patch(
            "torch_batteries.callbacks.model_checkpoint.logger.warning"
        ) as mock_warning:
            checkpoint._delete_saved_model(str(missing_path))  # noqa: SLF001

        assert str(missing_path) not in checkpoint.best_k_models
        mock_warning.assert_called_once_with(
            "Checkpoint file was already missing during cleanup: %s",
            str(missing_path),
        )

    def test_missing_monitor_metric_logs_warning(self, tmp_path: Path) -> None:
        """Missing checkpoint monitor data is visible at WARNING level."""
        checkpoint = ModelCheckpoint(
            stage="val", metric="accuracy", save_dir=str(tmp_path)
        )

        with patch(
            "torch_batteries.callbacks.model_checkpoint.logger.warning"
        ) as mock_warning:
            checkpoint.run_on_validation_end(
                {
                    "model": torch.nn.Linear(1, 1),
                    "val_metrics": {"loss": 0.5},
                    "epoch": 1,
                }
            )

        mock_warning.assert_called_once_with(
            "Checkpoint monitor metric '%s' is missing; checkpoint was skipped.",
            "accuracy",
        )

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

        for epoch, loss in enumerate((0.5, 0.3, 0.4), start=1):
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

    def test_checkpoint_save_log_is_unconditional(self, tmp_path: Path) -> None:
        """A successful checkpoint always emits its INFO outcome log."""
        checkpoint = ModelCheckpoint(
            stage="val", metric="accuracy", save_dir=str(tmp_path)
        )

        with patch(
            "torch_batteries.callbacks.model_checkpoint.logger.info"
        ) as mock_info:
            checkpoint.run_on_validation_end(
                {
                    "model": torch.nn.Linear(1, 1),
                    "val_metrics": {"accuracy": 0.8},
                    "epoch": 1,
                }
            )

        assert checkpoint.best_model_path is not None
        mock_info.assert_called_once_with(
            "Saved model checkpoint at: %s with %s: %.2f",
            checkpoint.best_model_path,
            "accuracy",
            0.8,
        )

    @pytest.mark.parametrize("mode", ["min", "max"])
    def test_state_round_trip_restores_checkpoint_ranking(self, mode: str) -> None:
        """Checkpoint ranking metadata survives callback serialization."""
        state = valid_checkpoint_state()
        if mode == "min":
            state.update(
                {
                    "best_model_path": "first.pth",
                    "best_score": 0.8,
                    "kth_best_model_path": "second.pth",
                    "kth_best_score": 0.9,
                }
            )
        source = ModelCheckpoint(
            stage="val",
            metric="score",
            mode=mode,  # type: ignore[arg-type]
        )
        source.load_state_dict(state)
        restored = ModelCheckpoint(
            stage="val",
            metric="score",
            mode=mode,  # type: ignore[arg-type]
        )

        restored.load_state_dict(source.state_dict())

        assert restored.best_k_models == {
            "first.pth": 0.8,
            "second.pth": 0.9,
        }
        assert restored.best_model_path == state["best_model_path"]
        assert restored.best_score == state["best_score"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("best_k_models", []),
            ("best_k_models", {"model.pth": "high"}),
            ("best_model_path", 1),
            ("kth_best_model_path", object()),
            ("best_score", "high"),
            ("kth_best_score", None),
            ("save_weights_only", True),
        ],
    )
    def test_invalid_state_is_rejected(self, field: str, value: object) -> None:
        """Every serialized ranking field is validated before restoration."""
        state = valid_checkpoint_state()
        state[field] = value
        checkpoint = ModelCheckpoint(stage="val", metric="score")

        with pytest.raises(
            ValueError, match="Invalid ModelCheckpoint checkpoint state"
        ):
            checkpoint.load_state_dict(state)

    def test_missing_state_field_is_rejected(self) -> None:
        """Incomplete callback state fails with the stable callback error."""
        checkpoint = ModelCheckpoint(stage="val", metric="score")

        with pytest.raises(
            ValueError, match="Invalid ModelCheckpoint checkpoint state"
        ):
            checkpoint.load_state_dict({})

    def test_stage_handlers_ignore_the_opposite_stage(self) -> None:
        """A checkpoint callback only handles its configured phase."""
        train_checkpoint = ModelCheckpoint(stage="train", metric="loss")
        val_checkpoint = ModelCheckpoint(stage="val", metric="loss")

        train_checkpoint.run_on_validation_end({})
        val_checkpoint.run_on_train_epoch_end({})

        assert train_checkpoint.best_model_path is None
        assert val_checkpoint.best_model_path is None

    def test_static_top_k_template_without_suffix_gets_unique_paths(
        self, tmp_path: Path
    ) -> None:
        """A suffixless static template gains an epoch before `.pth`."""
        checkpoint = ModelCheckpoint(
            stage="val",
            metric="accuracy",
            save_dir=str(tmp_path),
            save_path="best",
            save_top_k=2,
            save_weights_only=True,
        )
        model = torch.nn.Linear(1, 1)

        for epoch, accuracy in enumerate((0.8, 0.9), start=1):
            checkpoint.run_on_validation_end(
                {
                    "model": model,
                    "val_metrics": {"accuracy": accuracy},
                    "epoch": epoch,
                }
            )

        assert {Path(path).name for path in checkpoint.best_k_models} == {
            "best-epoch=1.pth",
            "best-epoch=2.pth",
        }

    def test_checkpoint_name_supports_prefix_without_metric_labels(self) -> None:
        """Internal filename formatting honors prefix and label controls."""
        checkpoint = ModelCheckpoint(stage="val", metric="accuracy")

        name = checkpoint._format_checkpoint_name(  # noqa: SLF001
            "{epoch}-{accuracy:.2f}",
            {"epoch": 2, "accuracy": 0.95},
            prefix="best",
            auto_insert_metric_name=False,
        )

        assert name == "best-2-0.95"
