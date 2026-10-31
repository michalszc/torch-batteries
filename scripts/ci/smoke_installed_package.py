"""Exercise an installed wheel on synthetic CPU data, including a spawned worker."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import torch_batteries
from torch_batteries import (
    Battery,
    CollectedMetric,
    Event,
    EventContext,
    StepOutput,
    charge,
)
from torch_batteries.callbacks import ModelCheckpoint

if TYPE_CHECKING:
    from torch_batteries.utils.metrics.types import MetricDefinition


class SmokeModel(nn.Module):
    """Tiny regression model whose event handlers use only the public API."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    @charge(Event.TRAIN_STEP)
    @charge(Event.VALIDATION_STEP)
    @charge(Event.TEST_STEP)
    def step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        predictions = cast("torch.Tensor", self.linear(inputs))
        return StepOutput(
            loss=nn.functional.mse_loss(predictions, targets),
            predictions=predictions,
            targets=targets,
        )

    @charge(Event.PREDICT_STEP)
    def predict(self, context: EventContext) -> torch.Tensor:
        inputs, _ = context["batch"]
        return cast("torch.Tensor", self.linear(inputs))


def _absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return a sample-weighted batch error."""
    return (predictions - targets).abs().mean()


def _phase_range(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute a metric requiring the complete phase's predictions."""
    del targets
    return predictions.max() - predictions.min()


def _battery(directory: Path) -> Battery:
    """Recreate matching checkpoint participants without optional dependencies."""
    model = SmokeModel()
    callback = ModelCheckpoint(
        phase="validation",
        metric="mae",
        mode="min",
        save_top_k=2,
        save_dir=str(directory),
    )
    metrics: dict[str, MetricDefinition] = {
        "mae": _absolute_error,
        "range": CollectedMetric(_phase_range),
    }
    return Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        callbacks=[callback],
        metrics=metrics,
    )


def exercise_workflows(directory: Path) -> None:
    """Assert training, evaluation, prediction, and full-state resume work together.

    Args:
        directory: Temporary checkpoint directory for this smoke run.
    """
    torch.manual_seed(7)
    inputs = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(inputs, inputs * 2)
    loader = DataLoader(dataset, batch_size=2)
    spawned_loader = DataLoader(
        dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn"
    )
    battery = _battery(directory)
    history = battery.fit(spawned_loader, loader, epochs=1, verbose=0)
    assert history["epochs_completed"] == 1
    assert torch.isfinite(torch.tensor(history["train_loss"])).all()
    assert len(history["val_metrics"]["range"]) == 1
    validation = battery.validate(loader, verbose=0)
    evaluation = battery.test(loader, verbose=0)
    assert torch.isfinite(torch.tensor(validation["val_loss"]))
    assert torch.isfinite(torch.tensor(evaluation["test_loss"]))
    predictions = battery.predict(loader, verbose=0, move_to_cpu=True, concatenate=True)
    assert predictions["predictions"].shape == (4, 1)
    checkpoint = directory / "explicit.pth"
    battery.save_checkpoint(checkpoint)
    restored = _battery(directory)
    restored.load_checkpoint(checkpoint)
    for expected, actual in zip(
        battery.model.parameters(), restored.model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
    resumed = restored.fit(loader, loader, epochs=2, verbose=0)
    assert resumed["epochs_completed"] == 2
    assert len(resumed["train_loss"]) == 2
    assert resumed["optimizer_steps"] > history["optimizer_steps"]


def main() -> None:
    """Require an installed package, then run workflows outside the source tree."""
    package = Path(torch_batteries.__file__).resolve().parent
    source = Path(__file__).resolve().parents[2] / "src"
    assert not package.is_relative_to(source), f"Imported source checkout: {package}"
    assert (package / "py.typed").is_file()
    with TemporaryDirectory(prefix="torch-batteries-wheel-") as directory:
        exercise_workflows(Path(directory))
    print(f"Installed torch-batteries {torch_batteries.__version__} smoke test passed")


if __name__ == "__main__":
    main()
