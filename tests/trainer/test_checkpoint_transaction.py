"""Tests for transactional checkpoint restoration."""

import copy
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from torch_batteries import Battery, DataPack
from torch_batteries.callbacks import Callback, GradientAccumulation
from torch_batteries.trainer._checkpoint import CheckpointMixin


class _CheckpointModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value]))
        self.register_buffer("running", torch.tensor([value + 0.5]))


class _CheckpointOptimizer(torch.optim.SGD):
    pass


class _CheckpointCallback(Callback):
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.value = int(state_dict["value"])


class _CheckpointMetric:
    def __init__(self, value: int) -> None:
        self.value = value

    def reset(self) -> None:
        self.value = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        del predictions, targets
        self.value += 1

    def compute(self) -> float:
        return float(self.value)

    def state_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.value = int(state_dict["value"])


class _CheckpointDataPack(DataPack):
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.value = int(state_dict["value"])


def _battery(
    value: int,
) -> tuple[
    Battery,
    _CheckpointOptimizer,
    _CheckpointCallback,
    _CheckpointMetric,
    _CheckpointDataPack,
]:
    model = _CheckpointModel(float(value))
    optimizer = _CheckpointOptimizer(model.parameters(), lr=0.01, momentum=0.9)
    optimizer.state[model.weight]["momentum_buffer"] = torch.tensor([value + 1.0])
    callback = _CheckpointCallback(value)
    metric = _CheckpointMetric(value)
    data_pack = _CheckpointDataPack(value)
    battery = Battery(
        model,
        device="cpu",
        optimizer=optimizer,
        callbacks=[callback],
        metrics={"metric": metric},
        data_pack=data_pack,
    )
    battery._last_completed_epoch = value  # noqa: SLF001
    battery._optimizer_step_idx = value + 1  # noqa: SLF001
    battery._train_results = {  # noqa: SLF001
        "train_loss": [float(value)],
        "val_loss": [float(value + 1)],
        "train_metrics": {"metric": [float(value)]},
        "val_metrics": {},
    }
    battery._loader_generator_states = {  # noqa: SLF001
        "train": {"loader.generator": torch.tensor([value], dtype=torch.uint8)}
    }
    battery._pending_loader_generator_states = copy.deepcopy(  # noqa: SLF001
        battery._loader_generator_states  # noqa: SLF001
    )
    battery._resume_loaded = value % 2 == 0  # noqa: SLF001
    battery._stop_training = value % 2 == 1  # noqa: SLF001
    return battery, optimizer, callback, metric, data_pack


def _state(
    battery: Battery,
    optimizer: _CheckpointOptimizer,
    callback: _CheckpointCallback,
    metric: _CheckpointMetric,
    data_pack: _CheckpointDataPack,
) -> dict[str, Any]:
    return copy.deepcopy(
        {
            "model": battery.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "callback": callback.state_dict(),
            "metric": metric.state_dict(),
            "data_pack": data_pack.state_dict(),
            "last_completed_epoch": battery._last_completed_epoch,  # noqa: SLF001
            "optimizer_step_idx": battery._optimizer_step_idx,  # noqa: SLF001
            "results": battery._train_results,  # noqa: SLF001
            "loader_generator_states": battery._loader_generator_states,  # noqa: SLF001
            "pending_loader_generator_states": (
                battery._pending_loader_generator_states  # noqa: SLF001
            ),
            "resume_loaded": battery._resume_loaded,  # noqa: SLF001
            "stop_training": battery._stop_training,  # noqa: SLF001
        }
    )


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def _fail_once_after(method: Callable[..., Any]) -> Callable[..., Any]:
    failed = False

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        result = method(*args, **kwargs)
        if not failed:
            failed = True
            message = "injected checkpoint load failure"
            raise RuntimeError(message)
        return result

    return wrapper


@contextmanager
def _inject_failure(
    point: str,
) -> Iterator[None]:
    target: Any
    method: Callable[..., Any]
    if point == "model":
        target = _CheckpointModel
        method_name = "load_state_dict"
        method = target.load_state_dict
    elif point == "optimizer":
        target = _CheckpointOptimizer
        method_name = "load_state_dict"
        method = target.load_state_dict
    elif point == "callback":
        target = _CheckpointCallback
        method_name = "load_state_dict"
        method = target.load_state_dict
    elif point == "metric":
        target = _CheckpointMetric
        method_name = "load_state_dict"
        method = target.load_state_dict
    elif point == "data_pack":
        target = _CheckpointDataPack
        method_name = "load_state_dict"
        method = target.load_state_dict
    elif point == "internal":
        target = CheckpointMixin
        method_name = "_apply_checkpoint_internal_state"
        method = target._apply_checkpoint_internal_state  # noqa: SLF001
    elif point == "rng":
        original = CheckpointMixin._restore_global_rng_state  # noqa: SLF001
        with patch.object(
            CheckpointMixin,
            "_restore_global_rng_state",
            side_effect=_fail_once_after(original),
        ):
            yield
        return
    else:
        raise AssertionError(point)

    with patch.object(target, method_name, _fail_once_after(method)):
        yield


@pytest.mark.parametrize(
    "failure_point",
    ["model", "optimizer", "callback", "metric", "data_pack", "internal", "rng"],
)
def test_failed_full_restore_rolls_back_every_component_and_rng(
    tmp_path: Path,
    failure_point: str,
) -> None:
    source, *_ = _battery(7)
    checkpoint = tmp_path / f"{failure_point}.pth"
    source.save_checkpoint(checkpoint)
    target_parts = _battery(2)
    target = target_parts[0]
    expected = _state(*target_parts)
    random.seed(101)
    torch.manual_seed(102)
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()

    with (
        _inject_failure(failure_point),
        pytest.raises(RuntimeError, match="injected checkpoint load failure"),
    ):
        target.load_checkpoint(checkpoint)

    _assert_nested_equal(_state(*target_parts), expected)
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_rollback_failure_is_logged_without_replacing_original_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source, *_ = _battery(7)
    checkpoint = tmp_path / "rollback-failure.pth"
    source.save_checkpoint(checkpoint)
    target_parts = _battery(2)
    target = target_parts[0]
    callback_method = _CheckpointCallback.load_state_dict
    model_method = _CheckpointModel.load_state_dict
    model_calls = 0

    def fail_model_rollback(*args: Any, **kwargs: Any) -> Any:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 2:
            message = "rollback failure"
            raise RuntimeError(message)
        return model_method(*args, **kwargs)

    with (
        patch.object(_CheckpointModel, "load_state_dict", fail_model_rollback),
        patch.object(
            _CheckpointCallback,
            "load_state_dict",
            _fail_once_after(callback_method),
        ),
        pytest.raises(RuntimeError, match="injected checkpoint load failure"),
    ):
        target.load_checkpoint(checkpoint)

    assert "Checkpoint rollback failed for model" in caplog.text


def test_raw_model_restore_is_transactional(tmp_path: Path) -> None:
    source = _CheckpointModel(9.0)
    checkpoint = tmp_path / "raw.pth"
    torch.save(source.state_dict(), checkpoint)
    target = _CheckpointModel(2.0)
    battery = Battery(target, device="cpu")
    expected = copy.deepcopy(target.state_dict())

    with (
        patch.object(
            _CheckpointModel,
            "load_state_dict",
            _fail_once_after(_CheckpointModel.load_state_dict),
        ),
        pytest.raises(RuntimeError, match="injected checkpoint load failure"),
    ):
        battery.load_checkpoint(checkpoint)

    _assert_nested_equal(target.state_dict(), expected)


def test_fixed_callback_configuration_is_validated_before_model_mutation(
    tmp_path: Path,
) -> None:
    source_model = _CheckpointModel(7.0)
    source = Battery(
        source_model,
        optimizer=_CheckpointOptimizer(source_model.parameters(), lr=0.1),
        callbacks=[GradientAccumulation(2)],
    )
    checkpoint = tmp_path / "callback-configuration.pth"
    source.save_checkpoint(checkpoint)
    target_model = _CheckpointModel(2.0)
    target = Battery(
        target_model,
        optimizer=_CheckpointOptimizer(target_model.parameters(), lr=0.1),
        callbacks=[GradientAccumulation(3)],
    )
    model_loads = 0
    model_method = _CheckpointModel.load_state_dict

    def record_model_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal model_loads
        model_loads += 1
        return model_method(*args, **kwargs)

    with (
        patch.object(_CheckpointModel, "load_state_dict", record_model_load),
        pytest.raises(ValueError, match="steps do not match"),
    ):
        target.load_checkpoint(checkpoint)

    assert model_loads == 0


def test_rejects_non_mapping_callback_snapshot_before_mutation(
    tmp_path: Path,
) -> None:
    source, *_ = _battery(7)
    checkpoint = tmp_path / "callback-snapshot.pth"
    source.save_checkpoint(checkpoint)
    target = _battery(2)[0]

    with (
        patch.object(_CheckpointCallback, "state_dict", return_value=[]),
        pytest.raises(TypeError, match=r"Callback state_dict\(\) must return"),
    ):
        target.load_checkpoint(checkpoint)


def test_rejects_non_mapping_data_pack_snapshot_before_mutation(
    tmp_path: Path,
) -> None:
    source, *_ = _battery(7)
    checkpoint = tmp_path / "data-pack-snapshot.pth"
    source.save_checkpoint(checkpoint)
    target = _battery(2)[0]

    with (
        patch.object(_CheckpointDataPack, "state_dict", return_value=[]),
        pytest.raises(TypeError, match=r"DataPack state_dict\(\) must return"),
    ):
        target.load_checkpoint(checkpoint)
