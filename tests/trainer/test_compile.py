"""Compatibility tests for models compiled before Battery construction."""

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, StepOutput, charge


class _CompiledModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)  # type: ignore[no-any-return]

    def _step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        predictions = self.forward(inputs)
        return StepOutput(
            loss=nn.functional.mse_loss(predictions, targets),
            predictions=predictions,
            targets=targets,
        )

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        return self._step(context)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self._step(context)

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        return self._step(context)

    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> torch.Tensor:
        inputs, _ = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return self.forward(inputs)


@pytest.fixture(autouse=True)
def reset_compiler_state() -> Iterator[None]:
    """Keep compiler caches isolated across compatibility tests."""
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _loader() -> DataLoader:
    inputs = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    targets = inputs.sum(dim=1, keepdim=True)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


def _mean_absolute_error(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return torch.mean(torch.abs(predictions - targets))


def _compiled_battery(*, backend: str | None = "eager") -> Battery:
    model = _CompiledModel()
    compiled = cast(
        "nn.Module",
        torch.compile(model)
        if backend is None
        else torch.compile(model, backend=backend),
    )
    optimizer = torch.optim.SGD(compiled.parameters(), lr=0.05)
    return Battery(
        compiled,
        device="cpu",
        optimizer=optimizer,
        metrics={"mae": _mean_absolute_error},
    )


def test_eager_backend_supports_all_workflows_and_metrics() -> None:
    """The eager compiler wrapper retains charged methods across workflows."""
    battery = _compiled_battery()
    loader = _loader()

    optimizer_parameters = [
        parameter
        for group in cast("torch.optim.Optimizer", battery.optimizer).param_groups
        for parameter in group["params"]
    ]
    assert all(
        any(
            parameter is optimizer_parameter
            for optimizer_parameter in optimizer_parameters
        )
        for parameter in battery.model.parameters()
    )

    fit_result = battery.fit(loader, loader, epochs=1, verbose=0)
    validation_result = battery.validate(loader, verbose=0)
    test_result = battery.test(loader, verbose=0)
    prediction_result = battery.predict(loader, verbose=0, concatenate=True)
    streamed = list(battery.predict_iter(loader, verbose=0))

    assert len(fit_result["train_metrics"]["mae"]) == 1
    assert len(fit_result["val_metrics"]["mae"]) == 1
    assert "mae" in validation_result["val_metrics"]
    assert "mae" in test_result["test_metrics"]
    assert prediction_result["predictions"].shape == (4, 1)
    assert len(streamed) == 2


def test_eager_backend_checkpoint_round_trip(tmp_path: Path) -> None:
    """Equivalent compiler wrappers accept complete resumable checkpoints."""
    source = _compiled_battery()
    source.train(_loader(), verbose=0)
    checkpoint = tmp_path / "compiled.pth"
    source.save_checkpoint(checkpoint)
    restored = _compiled_battery()

    restored.load_checkpoint(checkpoint)

    for source_value, restored_value in zip(
        source.model.state_dict().values(),
        restored.model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(source_value, restored_value)
    assert restored.optimizer is not None
    assert restored.optimizer.state_dict() == source.optimizer.state_dict()  # type: ignore[union-attr]


def test_default_backend_compiles_during_cpu_training() -> None:
    """The installed default backend executes an actual compiled training step."""
    battery = _compiled_battery(backend=None)

    result = battery.train(_loader(), verbose=0)

    assert len(result["train_loss"]) == 1
    assert len(result["train_metrics"]["mae"]) == 1
