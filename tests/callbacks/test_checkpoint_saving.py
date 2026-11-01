"""Checkpoint saving policies exercised through callback and training workflows."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.callbacks.test_checkpoint_ranking import make_battery, save
from torch_batteries import Battery, Event, EventContext, StepOutput, charge
from torch_batteries.callbacks import ModelCheckpoint


class EpochScoreModel(torch.nn.Module):
    """Supply deterministic epoch metrics while performing ordinary optimization."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def _step(self, context: EventContext) -> StepOutput:
        inputs, targets = context["batch"]
        predictions = self.layer(inputs)
        scores = (0.80, 0.79, 0.90, 0.85, 0.85, 0.70, 0.95, 0.88)
        return StepOutput(
            loss=((predictions - targets) ** 2).mean(),
            metrics={"score": scores[context["epoch"] - 1]},
        )

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        return self._step(context)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self._step(context)


def make_training_battery(
    directory: Path, *, extra_callbacks: list[object] | None = None, **options: Any
) -> tuple[Battery, ModelCheckpoint]:
    """Construct a reproducible CPU workflow with configurable saving policies."""
    configuration = {"phase": "validation", "metric": "score", **options}
    callback = ModelCheckpoint(save_dir=str(directory), **configuration)
    model = EpochScoreModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        callbacks=[callback, *(extra_callbacks or [])],
    )
    return battery, callback


def loader() -> DataLoader:
    """Create a one-batch loader with a checkpointable generator."""
    return DataLoader(
        TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)),
        batch_size=2,
        generator=torch.Generator().manual_seed(12),
    )


def assert_manifest_matches(callback: ModelCheckpoint, directory: Path) -> None:
    """Compare every serialized callback field with the committed manifest."""
    manifest = next(directory.glob(".torch-batteries-top-k-*.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload == {"schema_version": 1, "state": callback.state_dict()}


@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("weights_only", [False, True])
def test_unlimited_retains_worse_scores_and_ties(
    tmp_path: Path, mode: str, *, weights_only: bool
) -> None:
    battery, callback = make_battery(
        tmp_path, mode=mode, top_k=-1, weights_only=weights_only
    )
    scores = (0.8, 0.79, 0.9, 0.85, 0.85)
    for epoch, score in enumerate(scores, 1):
        save(battery, callback, epoch, score)
        assert_manifest_matches(callback, tmp_path)
    assert list(callback.best_k_models.values()) == list(scores)
    assert len(list(tmp_path.glob("*.pth"))) == len(scores)
    state = callback.state_dict()
    assert state["best_score"] == (min(scores) if mode == "min" else max(scores))
    assert state["kth_best_score"] == (max(scores) if mode == "min" else min(scores))
    for path in callback.best_k_models:
        payload = torch.load(path, weights_only=True)
        assert ("__torch_batteries_checkpoint__" in payload) is not weights_only


@pytest.mark.parametrize("template", ["best", "best.pt"])
def test_unlimited_static_template_preserves_each_epoch(
    tmp_path: Path, template: str
) -> None:
    battery, callback = make_training_battery(
        tmp_path, save_top_k=-1, save_path=template
    )
    battery.fit(loader(), loader(), epochs=3, verbose=0)
    assert {Path(path).name for path in callback.best_k_models} == {
        f"best-epoch={epoch}{'.pt' if template.endswith('.pt') else '.pth'}"
        for epoch in range(1, 4)
    }
    assert_manifest_matches(callback, tmp_path)


@pytest.mark.parametrize("phase", ["train", "validation"])
def test_cadence_intersects_actual_validation_epochs(
    tmp_path: Path, phase: str
) -> None:
    battery, callback = make_training_battery(
        tmp_path, phase=phase, save_top_k=-1, save_every_n_epochs=3
    )
    result = battery.fit(
        loader(), loader(), epochs=6, verbose=0, validate_every_n_epochs=2
    )
    expected = {3, 6} if phase == "train" else {6}
    assert {Path(path).name for path in callback.best_k_models} == {
        f"epoch={epoch}.pth" for epoch in expected
    }
    assert len(result["val_loss"]) == 3
    assert_manifest_matches(callback, tmp_path)


@pytest.mark.parametrize("phase", ["train", "validation"])
def test_ineligible_epoch_does_not_access_missing_metrics(
    tmp_path: Path, phase: str
) -> None:
    _, callback = make_training_battery(tmp_path, phase=phase, save_every_n_epochs=2)
    context: EventContext = {"epoch": 1, "model": torch.nn.Linear(1, 1)}
    handler = (
        callback.run_on_train_epoch_end
        if phase == "train"
        else callback.run_on_validation_end
    )
    handler(context)
    assert not list(tmp_path.iterdir())
    context["epoch"] = 2
    context["train_metrics"] = {}
    context["val_metrics"] = {}
    with pytest.raises(ValueError, match="unavailable"):
        handler(context)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, "2"])
def test_invalid_checkpoint_cadence(tmp_path: Path, interval: object) -> None:
    with pytest.raises(ValueError, match="save_every_n_epochs"):
        make_training_battery(tmp_path, save_every_n_epochs=interval)


@pytest.mark.parametrize("top_k", [0, -2, True, 1.5])
def test_invalid_checkpoint_retention(tmp_path: Path, top_k: object) -> None:
    with pytest.raises(ValueError, match="save_top_k"):
        make_training_battery(tmp_path, save_top_k=top_k)


def test_unlimited_resume_preserves_absolute_cadence_and_retained_files(
    tmp_path: Path,
) -> None:
    options: dict[str, Any] = {"save_top_k": -1, "save_every_n_epochs": 2}
    battery, callback = make_training_battery(tmp_path, **options)
    battery.fit(loader(), loader(), epochs=4, verbose=0)
    original = {path: Path(path).read_bytes() for path in callback.best_k_models}
    restored, restored_callback = make_training_battery(tmp_path, **options)
    restored.load_checkpoint(tmp_path / "epoch=2.pth")
    assert restored_callback.state_dict() == callback.state_dict()
    restored.fit(loader(), loader(), epochs=6, verbose=0)
    assert all(Path(path).read_bytes() == content for path, content in original.items())
    assert {Path(path).name for path in restored_callback.best_k_models} == {
        "epoch=2.pth",
        "epoch=4.pth",
        "epoch=4-v1.pth",
        "epoch=6.pth",
    }
    assert_manifest_matches(restored_callback, tmp_path)


def test_legacy_cadence_metadata_defaults_to_one(tmp_path: Path) -> None:
    battery, callback = make_training_battery(tmp_path)
    battery.fit(loader(), loader(), verbose=0)
    checkpoint = Path(callback.best_model_path)  # type: ignore[arg-type]
    payload = torch.load(checkpoint, weights_only=True)
    payload["callbacks"][0]["state"]["ranking_config"].pop("save_every_n_epochs", None)
    torch.save(payload, checkpoint)
    manifest = next(tmp_path.glob("*.json"))
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["state"]["ranking_config"].pop("save_every_n_epochs", None)
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    restored, restored_callback = make_training_battery(tmp_path)
    restored.load_checkpoint(checkpoint)
    assert restored_callback.state_dict() == callback.state_dict()
    incompatible, _ = make_training_battery(tmp_path, save_every_n_epochs=2)
    with pytest.raises(ValueError, match="Invalid ModelCheckpoint"):
        incompatible.load_checkpoint(checkpoint)


def test_unlimited_manifest_failure_preserves_files_and_state(tmp_path: Path) -> None:
    battery, callback = make_training_battery(tmp_path, save_top_k=-1)
    battery.fit(loader(), loader(), verbose=0)
    state = callback.state_dict()
    files = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with (
        patch.object(callback, "_write_manifest", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        callback.run_on_validation_end(
            {
                "battery": battery,
                "model": battery.model,
                "epoch": 2,
                "val_metrics": {"score": 0.7},
            }
        )
    assert callback.state_dict() == state
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == files
