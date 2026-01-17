"""Test for torch_batteries.callbacks.early_stopping module."""

from typing import TYPE_CHECKING

import pytest
import torch

from torch_batteries import Battery
from torch_batteries.callbacks.early_stopping import EarlyStopping

if TYPE_CHECKING:
    from torch_batteries.events import EventContext


class TestEarlyStopping:
    """Test cases for EarlyStopping callback."""

    def test_initialization(self) -> None:
        """Test EarlyStopping initialization with valid parameters."""
        early_stopping = EarlyStopping(
            stage="val",
            metric="loss",
            min_delta=0.01,
            patience=3,
            verbose=True,
            mode="min",
            restore_best_weights=True,
        )
        assert early_stopping._stage == "val"  # noqa: SLF001
        assert early_stopping._metric == "loss"  # noqa: SLF001
        assert early_stopping._min_delta == 0.01  # noqa: SLF001
        assert early_stopping._patience == 3  # noqa: SLF001
        assert early_stopping._verbose is True  # noqa: SLF001
        assert early_stopping._mode == "min"  # noqa: SLF001
        assert early_stopping._restore_best_weights is True  # noqa: SLF001

    def test_invalid_stage(self) -> None:
        """Test EarlyStopping initialization with invalid stage."""
        with pytest.raises(ValueError, match="stage must be one of 'train' or 'val'"):
            EarlyStopping(stage="invalid", metric="loss")  # type: ignore[arg-type]

    def test_invalid_mode(self) -> None:
        """Test EarlyStopping initialization with invalid mode."""
        with pytest.raises(ValueError, match="mode must be one of 'min' or 'max'"):
            EarlyStopping(stage="val", metric="loss", mode="invalid")  # type: ignore[arg-type]

    def test_run_on_train_start(self) -> None:
        """Test run_on_train_start method initializes parameters correctly."""
        early_stopping = EarlyStopping(stage="val", metric="loss")
        context: EventContext = {}
        early_stopping.run_on_train_start(context)
        assert early_stopping.best_score is None
        assert early_stopping._epochs_no_improve == 0  # noqa: SLF001

    def test_check_for_early_stop_min_mode(self) -> None:
        """Test _check_for_early_stop method in 'min' mode."""
        early_stopping = EarlyStopping(
            stage="val", metric="loss", mode="min", patience=2
        )
        model = torch.nn.Linear(1, 1)
        battery = Battery(model=model)

        context: EventContext = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 0.5},
        }

        early_stopping.run_on_validation_end(context)
        assert early_stopping.best_score == 0.5
        assert early_stopping._epochs_no_improve == 0  # noqa: SLF001

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 0.6},
        }

        early_stopping.run_on_validation_end(context)
        assert early_stopping._epochs_no_improve == 1  # noqa: SLF001

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 0.4},
        }
        early_stopping.run_on_validation_end(context)
        assert early_stopping.best_score == 0.4
        assert early_stopping._epochs_no_improve == 0  # noqa: SLF001

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 0.45},
        }
        early_stopping.run_on_validation_end(context)
        assert early_stopping._epochs_no_improve == 1  # noqa: SLF001

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 0.5},
        }
        early_stopping.run_on_validation_end(context)
        assert early_stopping._epochs_no_improve == 2  # noqa: SLF001

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 0.55},
        }
        early_stopping.run_on_validation_end(context)
        assert early_stopping._epochs_no_improve == 3  # noqa: SLF001
        assert early_stopping.best_score == 0.4

    def test_check_for_early_stop_max_mode(self) -> None:
        """Test _check_for_early_stop method in 'max' mode."""
        early_stopping = EarlyStopping(
            stage="val", metric="accuracy", mode="max", patience=2
        )
        model = torch.nn.Linear(1, 1)
        battery = Battery(model=model)

        context: EventContext = {
            "model": model,
            "battery": battery,
            "val_metrics": {"accuracy": 0.7},
        }
        early_stopping.run_on_validation_end(context)
        assert early_stopping.best_score == 0.7
        assert early_stopping._epochs_no_improve == 0  # noqa: SLF001

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"accuracy": 0.65},
        }
        early_stopping.run_on_validation_end(context)
        assert early_stopping._epochs_no_improve == 1  # noqa: SLF001

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"accuracy": 0.75},
        }
        early_stopping.run_on_validation_end(context)
        assert early_stopping.best_score == 0.75
        assert early_stopping._epochs_no_improve == 0  # noqa: SLF001

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"accuracy": 0.72},
        }
        early_stopping.run_on_validation_end(context)
        assert early_stopping._epochs_no_improve == 1  # noqa: SLF001

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"accuracy": 0.7},
        }
        early_stopping.run_on_validation_end(context)
        assert early_stopping._epochs_no_improve == 2  # noqa: SLF001

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"accuracy": 0.68},
        }
        early_stopping.run_on_validation_end(context)
        assert early_stopping._epochs_no_improve == 3  # noqa: SLF001
        assert early_stopping.best_score == 0.75

    def test_restore_best_weights(self) -> None:
        """Test that best weights are restored when restore_best_weights is True."""
        early_stopping = EarlyStopping(
            stage="val",
            metric="loss",
            mode="min",
            patience=1,
            restore_best_weights=True,
        )
        model = torch.nn.Linear(1, 1)
        battery = Battery(model=model)

        initial_weights = model.state_dict().copy()

        context: EventContext = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 0.5},
        }
        early_stopping.run_on_validation_end(context)

        for param in model.parameters():
            param.data += 1.0

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 0.6},
        }
        early_stopping.run_on_validation_end(context)

        for param in model.parameters():
            param.data += 1.0

        context = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 0.7},
        }
        early_stopping.run_on_validation_end(context)

        assert early_stopping.best_weights is not None

        for key in initial_weights:
            assert torch.equal(
                model.state_dict()[key], early_stopping.best_weights[key]
            )
