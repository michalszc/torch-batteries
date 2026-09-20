"""Filesystem tracker behavior and callback integration."""

import csv
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml  # type: ignore[import-untyped]
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, StepOutput, charge
from torch_batteries.callbacks import ExperimentTrackingCallback
from torch_batteries.tracking import LocalTracker, Run


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    @charge(Event.TRAIN_STEP)
    def train_step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return StepOutput(loss=nn.functional.mse_loss(self.linear(inputs), targets))

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return StepOutput(loss=nn.functional.mse_loss(self.linear(inputs), targets))


def test_local_tracker_records_callback_run(tmp_path: Path) -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    tracker = LocalTracker("model_a", save_dir=tmp_path)
    callback = ExperimentTrackingCallback(
        tracker, Run(name="trial", config={"optimizer": "custom", "batch_size": 2})
    )
    battery = Battery(model, optimizer=optimizer, callbacks=[callback], device="cpu")
    inputs = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    loader = DataLoader(TensorDataset(inputs, inputs), batch_size=2)

    battery.fit(loader, loader, epochs=2, verbose=0)

    run_dir = tmp_path / "model_a" / "version_0"
    assert tracker.run_dir == run_dir
    assert tracker.is_initialized is False
    hparams = yaml.safe_load((run_dir / "hparams.yaml").read_text())
    assert hparams["name"] == "trial"
    assert hparams["parameter_count"] == 2
    assert "Linear" in hparams["model"]
    assert hparams["optimizer"] == "custom"
    assert hparams["optimizer_settings"]["lr"] == 0.05
    assert hparams["batch_size"] == 2
    with (run_dir / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert list(rows[0]) == ["epoch", "train/loss", "val/loss"]
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert all(row["train/loss"] and row["val/loss"] for row in rows)
    summary = yaml.safe_load((run_dir / "summary.yaml").read_text())
    assert summary["exit_code"] == 0
    assert summary["total_epochs"] == 2
    assert isinstance(summary["train_loss"], float)
    assert isinstance(summary["val_loss"], float)
    assert not list(run_dir.glob("*.pt"))

    tracker.init(Run(config={"lr": 0.01}))
    assert tracker.run_dir == tmp_path / "model_a" / "version_1"
    tracker.finish()


def test_local_tracker_requires_active_run_and_valid_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_name"):
        LocalTracker("../bad", save_dir=tmp_path)
    tracker = LocalTracker("model_b", save_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not initialized"):
        tracker.log_metrics({"loss": 1.0})
    tracker.init(Run())
    with pytest.raises(RuntimeError, match="already initialized"):
        tracker.init(Run())
    tracker.log_metrics({"accuracy": 0.8}, step=3, prefix="val/")
    tracker.log_metrics({"loss": 0.2}, step=3, prefix="train/")
    tracker.log_metrics({"accuracy": 0.9}, step=4, prefix="val/")
    tracker.log_summary({"score": [0.3, 0.8], "metrics": {"accuracy": [0.7, 0.9]}})
    tracker.finish(exit_code=1)
    assert tracker.run_dir is not None
    with (tracker.run_dir / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"epoch": "3", "val/accuracy": "0.8", "train/loss": "0.2"},
        {"epoch": "4", "val/accuracy": "0.9", "train/loss": ""},
    ]
    assert yaml.safe_load((tracker.run_dir / "summary.yaml").read_text()) == {
        "exit_code": 1,
        "score": 0.8,
        "metrics": {"accuracy": 0.9},
    }
