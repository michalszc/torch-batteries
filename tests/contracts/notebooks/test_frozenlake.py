"""Execute the FrozenLake notebook definitions against Battery workflows."""

import ast
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from torch_batteries import Battery

NOTEBOOK = (
    Path(__file__).parents[3] / "notebooks" / "frozenlake_a2c_local_tracking.ipynb"
)
DEVICES = [
    "cpu",
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
]


@pytest.fixture
def definitions() -> dict[str, Any]:
    """Load actual notebook imports and classes, without application cells."""
    pytest.importorskip("gymnasium")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    names = {
        "FrozenLakeA2C",
        "OnPolicyRollout",
        "GreedyEvaluation",
        "FrozenLakeDataPack",
    }
    nodes: list[ast.stmt] = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or source.startswith("%"):
            continue
        for node in ast.parse(source).body:
            if (
                isinstance(node, (ast.Import, ast.ImportFrom))
                or (isinstance(node, ast.ClassDef) and node.name in names)
                or (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id in {"SEED", "logger"}
                        for target in node.targets
                    )
                )
            ):
                nodes.append(node)
    namespace: dict[str, Any] = {}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("device", DEVICES)
def test_training_and_evaluation_use_selected_device(
    definitions: dict[str, Any], device: str, tmp_path: Path
) -> None:
    """Lazy inference and subsequent optimization share Battery's device."""
    torch.manual_seed(17)
    model = definitions["FrozenLakeA2C"]()
    data_pack = definitions["FrozenLakeDataPack"](model)
    data_pack.rollout.episodes = 2
    data_pack.evaluation.episodes = 2
    tracker = definitions["LocalTracker"]("frozenlake", save_dir=tmp_path)
    callback = definitions["ExperimentTrackingCallback"](
        tracker, run=definitions["Run"](name="device-regression")
    )
    battery = Battery(
        model,
        device=device,
        optimizer=torch.optim.Adam(model.parameters(), lr=3e-3),
        data_pack=data_pack,
        callbacks=[callback],
    )
    observed_devices: list[torch.device] = []

    def inspect_input(_model: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        observed_devices.append(inputs[0].device)
        assert inputs[0].device == next(model.parameters()).device

    handle = model.register_forward_pre_hook(inspect_input)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    history = battery.train(epochs=2, verbose=0)
    assert history["optimizer_steps"] == 2
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))
    assert any(
        not torch.equal(original, parameter)
        for original, parameter in zip(before, model.parameters(), strict=True)
    )
    assert model.training
    assert data_pack.rollout.device == battery.device
    assert all(tensor.device.type == "cpu" for tensor in next(iter(data_pack.rollout)))
    rollout_calls = len(observed_devices)
    result = battery.test(verbose=0)
    assert len(observed_devices) > rollout_calls
    assert data_pack.evaluation.device == battery.device
    assert torch.isfinite(torch.tensor(result["test_loss"]))
    assert 0 <= result["test_metrics"]["success_rate"] <= 1
    assert data_pack.evaluation.frames
    assert next(iter(data_pack.evaluation)).device.type == "cpu"
    assert tracker.run_dir is not None
    assert (tracker.run_dir / "metrics.csv").is_file()
    handle.remove()


@pytest.mark.parametrize("dataset_name", ["OnPolicyRollout", "GreedyEvaluation"])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("failure", ["forward", "creation"])
def test_inference_failure_restores_mode_and_closes_environment(
    definitions: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    *,
    training: bool,
    failure: str,
) -> None:
    """Collection failures cannot leak an environment or alter model mode."""
    model = definitions["FrozenLakeA2C"]()
    model.train(training)
    environment = MagicMock()
    environment.reset.return_value = (0, {})
    maker = MagicMock(return_value=environment)
    if failure == "creation":
        maker.side_effect = RuntimeError("creation failed")
    else:
        monkeypatch.setattr(
            model, "forward", MagicMock(side_effect=RuntimeError("forward failed"))
        )
    monkeypatch.setattr(definitions["gym"], "make", maker)
    dataset = definitions[dataset_name](model, episodes=1)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        next(iter(dataset))
    assert model.training is training
    if failure == "forward":
        environment.close.assert_called_once_with()
    else:
        environment.close.assert_not_called()


@pytest.mark.parametrize("dataset_name", ["OnPolicyRollout", "GreedyEvaluation"])
def test_state_table_is_batched_and_refreshed_each_iteration(
    definitions: dict[str, Any], monkeypatch: pytest.MonkeyPatch, dataset_name: str
) -> None:
    """Every collection uses one fresh forward pass over all sixteen states."""
    model = definitions["FrozenLakeA2C"]()
    environment = MagicMock()
    environment.reset.return_value = (0, {})
    environment.step.return_value = (15, 1.0, True, False, {})
    monkeypatch.setattr(definitions["gym"], "make", MagicMock(return_value=environment))
    dataset = definitions[dataset_name](model, episodes=3)
    inputs_seen: list[torch.Tensor] = []

    def inspect_input(_model: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        inputs_seen.append(inputs[0].detach().clone())

    handle = model.register_forward_pre_hook(inspect_input)
    first = next(iter(dataset))
    assert len(inputs_seen) == 1
    assert torch.equal(inputs_seen[0], torch.eye(16))
    if dataset_name == "OnPolicyRollout":
        with torch.no_grad():
            model.critic.bias.add_(7)
    else:
        with torch.no_grad():
            model.actor.weight.zero_()
            model.actor.bias.copy_(torch.tensor([-100.0, -100.0, 100.0, -100.0]))
    second = next(iter(dataset))
    assert len(inputs_seen) == 2
    assert model.training
    if dataset_name == "OnPolicyRollout":
        assert torch.allclose(second[3], first[3] - 7)
        assert all(tensor.device.type == "cpu" for tensor in second)
    else:
        assert all(call.args == (2,) for call in environment.step.call_args_list[-3:])
    handle.remove()
