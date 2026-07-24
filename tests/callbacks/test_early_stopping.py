"""Test for torch_batteries.callbacks.early_stopping module."""

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from torch_batteries import Battery
from torch_batteries.callbacks.early_stopping import EarlyStopping

if TYPE_CHECKING:
    from torch_batteries.events import EventContext


class ModelWithBuffer(torch.nn.Module):
    """Small model containing both parameters and a registered buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.register_buffer("running_total", torch.zeros(1))


class TestEarlyStopping:
    """Test cases for EarlyStopping callback."""

    def test_initialization(self) -> None:
        """Test EarlyStopping initialization with valid parameters."""
        early_stopping = EarlyStopping(
            stage="val",
            metric="loss",
            min_delta=0.01,
            patience=3,
            mode="min",
            restore_best_weights=True,
        )
        assert early_stopping._stage == "val"  # noqa: SLF001
        assert early_stopping._metric == "loss"  # noqa: SLF001
        assert early_stopping._min_delta == 0.01  # noqa: SLF001
        assert early_stopping._patience == 3  # noqa: SLF001
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

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_delta": -0.1},
            {"patience": -1},
        ],
    )
    def test_invalid_non_negative_configuration(
        self, kwargs: dict[str, float | int]
    ) -> None:
        """Patience and minimum delta cannot be negative."""
        with pytest.raises(ValueError, match="greater than or equal to zero"):
            EarlyStopping(stage="val", metric="loss", **kwargs)  # type: ignore[arg-type]

    def test_run_on_train_start(self) -> None:
        """Test run_on_train_start method initializes parameters correctly."""
        early_stopping = EarlyStopping(stage="val", metric="loss")
        context: EventContext = {}
        early_stopping.run_on_train_start(context)
        assert early_stopping.best_score is None
        assert early_stopping._epochs_no_improve == 0  # noqa: SLF001
        assert early_stopping.best_weights is None

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

        initial_weights = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

        context: EventContext = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 0.5},
        }
        early_stopping.run_on_validation_end(context)

        saved_weights = early_stopping.best_weights
        assert saved_weights is not None

        for param in model.parameters():
            param.data += 1.0

        for key, initial_weight in initial_weights.items():
            assert torch.equal(saved_weights[key], initial_weight.cpu())

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
        early_stopping.run_on_train_end({"model": model})

        for key, initial_weight in initial_weights.items():
            assert torch.equal(model.state_dict()[key], initial_weight)

    def test_snapshot_and_restore_include_buffers(self) -> None:
        """Best-state restoration includes model buffers as well as parameters."""
        model = torch.nn.BatchNorm1d(2)
        battery = Battery(model=model)
        early_stopping = EarlyStopping(
            stage="val", metric="loss", restore_best_weights=True
        )
        context: EventContext = {
            "model": model,
            "battery": battery,
            "val_metrics": {"loss": 1.0},
        }
        early_stopping.run_on_validation_end(context)
        assert model.running_mean is not None
        expected_mean = model.running_mean.detach().clone()
        model.running_mean.add_(5.0)

        early_stopping.run_on_train_end({"model": model})

        assert model.running_mean is not None
        assert torch.equal(model.running_mean, expected_mean)

    @pytest.mark.parametrize(
        "device",
        [
            "auto",
            pytest.param(
                "mps",
                marks=pytest.mark.skipif(
                    not torch.backends.mps.is_available(), reason="MPS is unavailable"
                ),
            ),
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is unavailable"
                ),
            ),
        ],
    )
    def test_snapshot_restores_state_without_changing_model_device(
        self, device: str
    ) -> None:
        """CPU snapshots restore parameters and buffers on their original device."""
        model = ModelWithBuffer()
        battery = Battery(model=model, device=device)
        original_parameter_devices = {
            name: parameter.device for name, parameter in model.named_parameters()
        }
        original_buffer_devices = {
            name: buffer.device for name, buffer in model.named_buffers()
        }
        expected_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        early_stopping = EarlyStopping(
            stage="val", metric="loss", restore_best_weights=True
        )
        early_stopping.run_on_validation_end(
            {
                "model": model,
                "battery": battery,
                "val_metrics": {"loss": 1.0},
            }
        )
        saved_weights = early_stopping.best_weights
        assert saved_weights is not None
        assert all(weight.device.type == "cpu" for weight in saved_weights.values())

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(5.0)
            for buffer in model.buffers():
                buffer.add_(5.0)
        early_stopping.run_on_train_end({"model": model})

        assert {
            name: parameter.device for name, parameter in model.named_parameters()
        } == original_parameter_devices
        assert {
            name: buffer.device for name, buffer in model.named_buffers()
        } == original_buffer_devices
        for key, expected_value in expected_state.items():
            assert torch.equal(model.state_dict()[key].cpu(), expected_value)

    def test_train_start_resets_saved_state(self) -> None:
        """Reusing a callback does not retain state from a previous run."""
        model = torch.nn.Linear(1, 1)
        battery = Battery(model=model)
        early_stopping = EarlyStopping(
            stage="val", metric="loss", restore_best_weights=True
        )
        early_stopping.run_on_validation_end(
            {"model": model, "battery": battery, "val_metrics": {"loss": 1.0}}
        )
        assert early_stopping.best_weights is not None

        early_stopping.run_on_train_start({})

        assert early_stopping.best_score is None
        assert early_stopping.best_weights is None

    def test_outcome_logs_are_unconditional(self) -> None:
        """Stop and restoration outcomes are logged without callback flags."""
        model = torch.nn.Linear(1, 1)
        battery = Battery(model=model)
        early_stopping = EarlyStopping(
            stage="val",
            metric="loss",
            patience=1,
            restore_best_weights=True,
        )

        with patch("torch_batteries.callbacks.early_stopping.logger.info") as mock_info:
            early_stopping.run_on_validation_end(
                {"model": model, "battery": battery, "val_metrics": {"loss": 1.0}}
            )
            early_stopping.run_on_validation_end(
                {"model": model, "battery": battery, "val_metrics": {"loss": 2.0}}
            )
            early_stopping.run_on_train_end({"model": model})

        assert mock_info.call_args_list[0].args == (
            "Early stopping applied. No improvement in '%s' for %d epochs.",
            "loss",
            1,
        )
        assert mock_info.call_args_list[1].args == (
            "Restored best model weights from early stopping.",
        )

    def test_state_round_trip_preserves_progress_and_weights(self) -> None:
        """Serialized state can be restored into a fresh callback."""
        model = torch.nn.Linear(1, 1)
        battery = Battery(model=model)
        source = EarlyStopping(stage="val", metric="loss", restore_best_weights=True)
        source.run_on_validation_end(
            {"model": model, "battery": battery, "val_metrics": {"loss": 0.5}}
        )
        source.run_on_validation_end(
            {"model": model, "battery": battery, "val_metrics": {"loss": 0.6}}
        )

        restored = EarlyStopping(stage="val", metric="loss", restore_best_weights=True)
        restored.load_state_dict(source.state_dict())

        assert restored.best_score == 0.5
        assert restored._epochs_no_improve == 1  # noqa: SLF001
        assert restored.best_weights is not None
        assert source.best_weights is not None
        for name, value in source.best_weights.items():
            assert torch.equal(restored.best_weights[name], value)

    @pytest.mark.parametrize(
        "state",
        [
            {},
            {
                "best_score": 1.0,
                "epochs_no_improve": object(),
                "best_weights": None,
            },
            {
                "best_score": 1.0,
                "epochs_no_improve": None,
                "best_weights": None,
            },
        ],
    )
    def test_invalid_state_is_rejected(self, state: dict[str, object]) -> None:
        """Malformed callback checkpoint data raises a stable public error."""
        callback = EarlyStopping(stage="val", metric="loss")

        with pytest.raises(ValueError, match="Invalid EarlyStopping checkpoint state"):
            callback.load_state_dict(state)

    def test_resumed_train_start_preserves_restored_state(self) -> None:
        """A resume event does not reset state loaded from a checkpoint."""
        callback = EarlyStopping(stage="val", metric="loss")
        callback.load_state_dict(
            {
                "best_score": 0.25,
                "epochs_no_improve": 3,
                "best_weights": None,
            }
        )

        callback.run_on_train_start({"resumed": True})

        assert callback.best_score == 0.25
        assert callback._epochs_no_improve == 3  # noqa: SLF001

    def test_stage_handlers_ignore_the_opposite_stage(self) -> None:
        """Only the configured phase can update early-stopping state."""
        model = torch.nn.Linear(1, 1)
        battery = Battery(model=model)
        train_callback = EarlyStopping(stage="train", metric="loss")
        val_callback = EarlyStopping(stage="val", metric="loss")

        train_callback.run_on_validation_end({})
        val_callback.run_on_epoch_end({})

        assert train_callback.best_score is None
        assert val_callback.best_score is None

        train_callback.run_on_epoch_end(
            {
                "model": model,
                "battery": battery,
                "train_metrics": {"loss": 0.5},
            }
        )
        assert train_callback.best_score == 0.5

    def test_missing_monitored_metric_is_rejected(self) -> None:
        """A configured metric must be present in the selected phase."""
        model = torch.nn.Linear(1, 1)
        callback = EarlyStopping(stage="val", metric="accuracy")

        with pytest.raises(ValueError, match="Metric 'accuracy' not found"):
            callback.run_on_validation_end(
                {
                    "model": model,
                    "battery": Battery(model=model),
                    "val_metrics": {"loss": 1.0},
                }
            )

    def test_later_improvement_replaces_best_weight_snapshot(self) -> None:
        """Best weights follow a later improvement rather than the baseline."""
        model = torch.nn.Linear(1, 1)
        battery = Battery(model=model)
        callback = EarlyStopping(stage="val", metric="loss", restore_best_weights=True)
        callback.run_on_validation_end(
            {"model": model, "battery": battery, "val_metrics": {"loss": 1.0}}
        )

        with torch.no_grad():
            model.weight.add_(2.0)
        expected = model.weight.detach().cpu().clone()
        callback.run_on_validation_end(
            {"model": model, "battery": battery, "val_metrics": {"loss": 0.5}}
        )

        assert callback.best_weights is not None
        assert torch.equal(callback.best_weights["weight"], expected)
