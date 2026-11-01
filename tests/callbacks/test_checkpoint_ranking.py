"""Regression coverage for durable top-k rankings and historical restoration."""

import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch

from torch_batteries import Battery
from torch_batteries.callbacks import ModelCheckpoint


def make_battery(
    directory: Path,
    *,
    mode: str = "max",
    top_k: int = 2,
    weights_only: bool = False,
) -> tuple[Battery, ModelCheckpoint]:
    """Create identically configured restoration participants."""
    callback = ModelCheckpoint(
        phase="validation",
        metric="score",
        mode=mode,  # type: ignore[arg-type]
        save_dir=str(directory),
        save_path="{epoch}.pth",
        save_top_k=top_k,
        save_weights_only=weights_only,
    )
    return Battery(torch.nn.Linear(1, 1), device="cpu", callbacks=[callback]), callback


def save(battery: Battery, callback: ModelCheckpoint, epoch: int, score: float) -> None:
    """Save a candidate with distinguishable model and epoch state."""
    with torch.no_grad():
        battery.model.weight.fill_(epoch)  # type: ignore[operator]
    battery._last_completed_epoch = epoch  # noqa: SLF001
    callback.run_on_validation_end(
        {
            "battery": battery,
            "model": battery.model,
            "epoch": epoch,
            "val_metrics": {"score": score},
        }
    )


@pytest.mark.parametrize(
    ("mode", "scores"),
    [("max", (0.80, 0.79, 0.90, 0.85)), ("min", (0.20, 0.21, 0.10, 0.15))],
)
def test_each_retained_checkpoint_restores_current_directory_ranking(
    tmp_path: Path, mode: str, scores: tuple[float, ...]
) -> None:
    battery, callback = make_battery(tmp_path, mode=mode)
    for epoch, score in enumerate(scores, 1):
        save(battery, callback, epoch, score)
    ranking = dict(callback.best_k_models)
    assert {Path(path).name for path in ranking} == {"epoch=3.pth", "epoch=4.pth"}
    for path in ranking:
        restored, restored_callback = make_battery(tmp_path, mode=mode)
        restored.load_checkpoint(path)
        epoch = int(Path(path).stem.split("=")[1])
        assert restored._last_completed_epoch == epoch  # noqa: SLF001
        assert restored.model.weight.item() == epoch  # type: ignore[operator]
        assert restored_callback.best_k_models == ranking
        assert restored_callback.best_model_path == callback.best_model_path
        assert restored_callback.best_score == scores[2]


def test_replayed_epoch_does_not_overwrite_a_better_retained_checkpoint(
    tmp_path: Path,
) -> None:
    battery, callback = make_battery(tmp_path)
    save(battery, callback, 1, 0.80)
    save(battery, callback, 2, 0.90)
    first = tmp_path / "epoch=1.pth"
    best = tmp_path / "epoch=2.pth"
    best_bytes = best.read_bytes()
    restored, restored_callback = make_battery(tmp_path)
    restored.load_checkpoint(first)
    save(restored, restored_callback, 2, 0.85)
    assert best.read_bytes() == best_bytes
    assert sorted(restored_callback.best_k_models.values()) == [0.85, 0.90]
    assert len(list(tmp_path.glob("*.pth"))) == 2
    assert not first.exists()


@pytest.mark.parametrize("damage", ["missing", "json", "schema", "config"])
def test_unavailable_manifest_falls_back_without_requiring_deleted_siblings(
    tmp_path: Path, damage: str, caplog: pytest.LogCaptureFixture
) -> None:
    battery, callback = make_battery(tmp_path)
    for epoch, score in enumerate((0.80, 0.79, 0.90, 0.85), 1):
        save(battery, callback, epoch, score)
    manifest = next(tmp_path.glob(".torch-batteries-top-k-*.json"))
    if damage == "missing":
        manifest.unlink()
    elif damage == "json":
        manifest.write_text("invalid", encoding="utf-8")
    else:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if damage == "schema":
            payload["schema_version"] = 999
        else:
            payload["state"]["ranking_config"]["metric"] = "other"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    restored, restored_callback = make_battery(tmp_path)
    restored.load_checkpoint(callback.best_model_path)  # type: ignore[arg-type]
    assert restored_callback.best_k_models == {callback.best_model_path: 0.90}
    assert "snapshot" in caplog.text
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_manifest_failure_preserves_checkpoint_files_and_ranking(
    tmp_path: Path,
) -> None:
    battery, callback = make_battery(tmp_path)
    save(battery, callback, 1, 0.8)
    state = callback.state_dict()
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with (
        patch.object(callback, "_write_manifest", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        save(battery, callback, 2, 0.9)
    assert callback.state_dict() == state
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_failed_restore_rolls_back_reconciled_ranking_without_disk_writes(
    tmp_path: Path,
) -> None:
    source, callback = make_battery(tmp_path)
    for epoch, score in enumerate((0.8, 0.79, 0.9, 0.85), 1):
        save(source, callback, epoch, score)
    target, target_callback = make_battery(tmp_path)
    previous_state = target_callback.state_dict()
    previous_weight = cast("torch.nn.Linear", target.model).weight.detach().clone()
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with (
        patch.object(
            Battery,
            "_restore_global_rng_state",
            side_effect=[RuntimeError("rng"), None],
        ),
        pytest.raises(RuntimeError, match="rng"),
    ):
        target.load_checkpoint(callback.best_model_path)  # type: ignore[arg-type]
    assert target_callback.state_dict() == previous_state
    assert torch.equal(target.model.weight, previous_weight)  # type: ignore[arg-type]
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("weights_only", [False, True])
def test_manifests_are_isolated_between_fresh_callbacks(
    tmp_path: Path, *, weights_only: bool
) -> None:
    first, first_callback = make_battery(tmp_path, weights_only=weights_only)
    second, second_callback = make_battery(tmp_path, weights_only=weights_only)
    save(first, first_callback, 1, 0.8)
    original_path = Path(next(iter(first_callback.best_k_models)))
    original = original_path.read_bytes()
    save(second, second_callback, 1, 0.9)
    assert original_path.read_bytes() == original
    assert len(list(tmp_path.glob(".torch-batteries-top-k-*.json"))) == 2
    assert set(first_callback.best_k_models).isdisjoint(second_callback.best_k_models)


def test_legacy_state_loads_exactly_and_reconciles_missing_paths(
    tmp_path: Path,
) -> None:
    battery, callback = make_battery(tmp_path)
    save(battery, callback, 1, 0.8)
    state = callback.state_dict()
    state.pop("ranking_id", None)
    state.pop("ranking_config", None)
    Path(callback.best_model_path).unlink()  # type: ignore[arg-type]
    restored, restored_callback = make_battery(tmp_path)
    restored_callback.load_state_dict(state)
    assert restored_callback.best_k_models == state["best_k_models"]
    restored_callback._reconcile_checkpoint_state()  # noqa: SLF001
    assert restored_callback.best_k_models == {}
    assert restored_callback.best_model_path is None
    assert restored_callback.best_score == float("-inf")
    save(restored, restored_callback, 2, 0.9)
    assert restored_callback.best_score == 0.9


@pytest.mark.parametrize("weights_only", [False, True])
def test_static_top_one_manifest_failure_restores_overwritten_file(
    tmp_path: Path, *, weights_only: bool
) -> None:
    model = torch.nn.Linear(1, 1)
    callback = ModelCheckpoint(
        phase="validation",
        metric="score",
        save_dir=str(tmp_path),
        save_path="best.pth",
        save_weights_only=weights_only,
    )
    battery = Battery(model, device="cpu", callbacks=[callback])
    save(battery, callback, 1, 0.8)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    state = callback.state_dict()
    with (
        patch.object(callback, "_write_manifest", side_effect=OSError("manifest")),
        pytest.raises(OSError, match="manifest"),
    ):
        save(battery, callback, 2, 0.9)
    assert callback.state_dict() == state
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before
    save(battery, callback, 2, 0.9)
    assert callback.best_model_path == str(tmp_path / "best.pth")
    assert len(list(tmp_path.glob("*.pth"))) == 1


def test_tied_candidate_leaves_full_ranking_and_manifest_unchanged(
    tmp_path: Path,
) -> None:
    battery, callback = make_battery(tmp_path)
    save(battery, callback, 1, 0.8)
    save(battery, callback, 2, 0.9)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    save(battery, callback, 3, 0.8)
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("ranking_id", [1, "invalid", "A" * 32])
def test_invalid_run_identity_is_rejected_before_mutation(
    tmp_path: Path, ranking_id: object
) -> None:
    _, callback = make_battery(tmp_path)
    before = callback.state_dict()
    corrupted = {**before, "ranking_id": ranking_id}
    with pytest.raises(ValueError, match="Invalid ModelCheckpoint"):
        callback.load_state_dict(corrupted)
    assert callback.state_dict() == before


def test_manifest_with_foreign_identity_is_not_used(tmp_path: Path) -> None:
    battery, callback = make_battery(tmp_path)
    save(battery, callback, 1, 0.8)
    manifest = next(tmp_path.glob(".torch-batteries-top-k-*.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["state"]["ranking_id"] = "0" * 32
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    target, target_callback = make_battery(tmp_path)
    target.load_checkpoint(callback.best_model_path)  # type: ignore[arg-type]
    assert target_callback.best_k_models == callback.best_k_models
