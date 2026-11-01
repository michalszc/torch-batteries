"""Final checkpoint files remain separate from periodic retention."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from tests.callbacks.test_checkpoint_saving import (
    assert_manifest_matches,
    loader,
    make_training_battery,
)
from tests.trainer.test_multi_dataset import NamedModel, SeededNamedPack
from torch_batteries import Battery, Event, EventContext, charge
from torch_batteries.callbacks import ModelCheckpoint


@pytest.mark.parametrize("phase", ["train", "validation"])
@pytest.mark.parametrize("weights_only", [False, True])
def test_final_save_is_additional_and_ignores_cadence(
    tmp_path: Path, phase: str, *, weights_only: bool
) -> None:
    battery, callback = make_training_battery(
        tmp_path,
        phase=phase,
        save_top_k=2,
        save_every_n_epochs=2,
        save_final=True,
        save_weights_only=weights_only,
    )
    assert callback.final_model_path is None
    result = battery.fit(loader(), loader(), epochs=3, verbose=0)
    assert {Path(path).name for path in callback.best_k_models} == {"epoch=2.pth"}
    final = Path(callback.final_model_path)  # type: ignore[arg-type]
    assert final.name == "final-epoch=3.pth"
    assert str(final) not in callback.best_k_models
    payload = torch.load(final, weights_only=True)
    weights = payload if weights_only else payload["model"]
    for key, tensor in battery.model.state_dict().items():
        assert torch.equal(weights[key], tensor)
    if not weights_only:
        assert payload["epoch"] == result["epochs_completed"] == 3
        assert payload["results"] == result
        assert payload["callbacks"][0]["state"] == callback.state_dict()
        assert battery.optimizer is not None
        assert payload["optimizer"] == battery.optimizer.state_dict()
        assert set(payload["loader_generator_states"]) == {"train", "validation"}
    assert_manifest_matches(callback, tmp_path)


@pytest.mark.parametrize("phase", ["train", "validation"])
@pytest.mark.parametrize("mode", ["min", "max"])
def test_final_before_any_periodic_checkpoint_has_valid_json(
    tmp_path: Path, phase: str, mode: str
) -> None:
    battery, callback = make_training_battery(
        tmp_path, phase=phase, mode=mode, save_final=True, save_every_n_epochs=10
    )
    result = battery.train(loader(), epochs=1, verbose=0)
    assert callback.best_k_models == {}
    assert callback.best_model_path is None
    assert_manifest_matches(callback, tmp_path)
    final = Path(callback.final_model_path)  # type: ignore[arg-type]
    payload = torch.load(final, weights_only=True)
    assert {key: payload["results"][key] for key in result} == result
    assert payload["results"]["val_loss"] == []
    restored, restored_callback = make_training_battery(
        tmp_path, phase=phase, mode=mode, save_final=True, save_every_n_epochs=10
    )
    restored.load_checkpoint(final)
    assert restored_callback.state_dict() == callback.state_dict()


@pytest.mark.parametrize(
    ("periodic", "final_template", "expected"),
    [
        (None, None, "final-epoch=1.pth"),
        ("best.pt", None, "final-best.pt"),
        ("models/{epoch}.pt", None, "models/final-epoch=1.pt"),
        ("best", None, "final-best.pth"),
        ("best.pt", "export/model", "export/model.pth"),
        (None, "export/{epoch}-{score:.2f}.bin", "export/epoch=1-score=0.80.bin"),
    ],
)
def test_final_filename_templates(
    tmp_path: Path, periodic: str | None, final_template: str | None, expected: str
) -> None:
    battery, callback = make_training_battery(
        tmp_path, save_path=periodic, final_save_path=final_template, save_final=True
    )
    battery.fit(loader(), loader(), verbose=0)
    assert callback.final_model_path == str(tmp_path / expected)
    assert (tmp_path / expected).is_file()
    assert_manifest_matches(callback, tmp_path)


def test_validation_final_template_uses_latest_available_validation_metrics(
    tmp_path: Path,
) -> None:
    battery, callback = make_training_battery(
        tmp_path, save_final=True, final_save_path="{epoch}-{score:.2f}.pt"
    )
    battery.fit(loader(), loader(), epochs=3, verbose=0, validate_every_n_epochs=2)
    assert callback.final_model_path == str(tmp_path / "epoch=3-score=0.79.pt")


def test_missing_final_template_metric_leaves_committed_state_untouched(
    tmp_path: Path,
) -> None:
    battery, callback = make_training_battery(
        tmp_path, save_final=True, final_save_path="{missing}.pt"
    )
    with pytest.raises(KeyError, match="missing"):
        battery.fit(loader(), loader(), verbose=0)
    assert callback.final_model_path is None
    assert len(list(tmp_path.glob("*.pth"))) == 1
    assert_manifest_matches(callback, tmp_path)


@pytest.mark.parametrize("workflow", ["train", "fit", "validate"])
def test_default_does_not_create_final_files(tmp_path: Path, workflow: str) -> None:
    battery, callback = make_training_battery(tmp_path)
    if workflow == "fit":
        battery.fit(loader(), loader(), verbose=0)
    else:
        getattr(battery, workflow)(loader(), verbose=0)
    assert callback.final_model_path is None
    assert not list(tmp_path.glob("final-*"))


def test_standalone_validation_does_not_create_or_replace_final(tmp_path: Path) -> None:
    battery, callback = make_training_battery(tmp_path, save_final=True)
    battery.validate(loader(), verbose=0)
    assert callback.final_model_path is None
    battery.fit(loader(), loader(), epochs=2, verbose=0)
    final = Path(callback.final_model_path)  # type: ignore[arg-type]
    before = final.read_bytes()
    battery.validate(loader(), verbose=0)
    assert final.read_bytes() == before
    assert callback.final_model_path == str(final)
    assert_manifest_matches(callback, tmp_path)


@pytest.mark.parametrize("template", [None, "last.pt"])
def test_repeated_training_keeps_only_latest_owned_final(
    tmp_path: Path, template: str | None
) -> None:
    battery, callback = make_training_battery(
        tmp_path, save_final=True, final_save_path=template
    )
    battery.fit(loader(), loader(), verbose=0)
    previous = Path(callback.final_model_path)  # type: ignore[arg-type]
    battery.fit(loader(), loader(), epochs=2, verbose=0)
    current = Path(callback.final_model_path)  # type: ignore[arg-type]
    assert current.is_file()
    assert previous == current or not previous.exists()
    assert set(tmp_path.glob("*.pth")) | set(tmp_path.glob("*.pt")) == {
        current,
        *(Path(path) for path in callback.best_k_models),
    }
    assert_manifest_matches(callback, tmp_path)


def test_final_destination_protects_unrelated_and_periodic_files(
    tmp_path: Path,
) -> None:
    unrelated = tmp_path / "best.pt"
    unrelated.write_bytes(b"unrelated")
    battery, callback = make_training_battery(
        tmp_path, save_final=True, save_path="best.pt", final_save_path="best.pt"
    )
    battery.fit(loader(), loader(), verbose=0)
    assert unrelated.read_bytes() == b"unrelated"
    assert callback.best_model_path == str(tmp_path / "best-v1.pt")
    assert callback.final_model_path == str(tmp_path / "best-v2.pt")
    # Periodic replacement must also preserve the previous final destination.
    final = Path(callback.final_model_path)
    before = final.read_bytes()
    callback.run_on_validation_end(
        {
            "battery": battery,
            "model": battery.model,
            "epoch": 2,
            "val_metrics": {"score": 1.0},
        }
    )
    assert final.read_bytes() == before
    assert_manifest_matches(callback, tmp_path)


@pytest.mark.parametrize("same_path", [False, True])
@pytest.mark.parametrize("failure", ["checkpoint", "manifest"])
def test_failed_final_replacement_preserves_state_and_files(
    tmp_path: Path, failure: str, *, same_path: bool
) -> None:
    battery, callback = make_training_battery(
        tmp_path, save_final=True, final_save_path="last.pt" if same_path else None
    )
    battery.fit(loader(), loader(), verbose=0)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    state = callback.state_dict()
    target = "_write_manifest" if failure == "manifest" else None
    mocked = (
        patch.object(callback, target, side_effect=OSError("disk full"))
        if target
        else patch.object(Battery, "save_checkpoint", side_effect=OSError("disk full"))
    )
    with mocked, pytest.raises(OSError, match="disk full"):
        callback.run_on_train_end(
            {"battery": battery, "model": battery.model, "epoch": 2}
        )
    assert callback.state_dict() == state
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_final_and_historical_restore_reconcile_latest_final_path(
    tmp_path: Path,
) -> None:
    options: dict[str, Any] = {"save_final": True, "save_top_k": 2}
    battery, callback = make_training_battery(tmp_path, **options)
    battery.fit(loader(), loader(), epochs=4, verbose=0)
    assert callback.final_model_path is not None
    for path in [*callback.best_k_models, callback.final_model_path]:
        restored, restored_callback = make_training_battery(tmp_path, **options)
        restored.load_checkpoint(path)
        assert restored_callback.state_dict() == callback.state_dict()
    old_final = Path(callback.final_model_path)
    restored.fit(loader(), loader(), epochs=6, verbose=0)
    assert not old_final.exists()
    assert restored_callback.final_model_path is not None
    assert Path(restored_callback.final_model_path).name == "final-epoch=6.pth"
    assert_manifest_matches(restored_callback, tmp_path)


@pytest.mark.parametrize("missing", ["final", "manifest"])
def test_final_path_reconciliation_handles_missing_files(
    tmp_path: Path, missing: str
) -> None:
    battery, callback = make_training_battery(tmp_path, save_final=True)
    battery.fit(loader(), loader(), verbose=0)
    final = Path(callback.final_model_path)  # type: ignore[arg-type]
    (final if missing == "final" else next(tmp_path.glob("*.json"))).unlink()
    restored, restored_callback = make_training_battery(tmp_path, save_final=True)
    assert callback.best_model_path is not None
    restored.load_checkpoint(callback.best_model_path)
    assert restored_callback.final_model_path is None


def test_legacy_final_metadata_defaults_and_invalid_state_are_transactional(
    tmp_path: Path,
) -> None:
    battery, callback = make_training_battery(tmp_path)
    battery.fit(loader(), loader(), verbose=0)
    state: dict[str, Any] = callback.state_dict()
    state.pop("final_model_path", None)
    for key in ("save_final", "final_save_path"):
        state["ranking_config"].pop(key, None)
    callback.load_state_dict(state)
    assert callback.final_model_path is None
    for key, value in (("final_model_path", 12), ("ranking_config", [])):
        before = callback.state_dict()
        with pytest.raises(ValueError, match="Invalid ModelCheckpoint"):
            callback.load_state_dict({**before, key: value})
        assert callback.state_dict() == before
    _, incompatible = make_training_battery(tmp_path, save_final=True)
    with pytest.raises(ValueError, match="Invalid ModelCheckpoint"):
        incompatible.load_state_dict(state)


def test_legacy_checkpoint_and_manifest_without_final_options_restore(
    tmp_path: Path,
) -> None:
    battery, callback = make_training_battery(tmp_path)
    battery.fit(loader(), loader(), verbose=0)
    assert callback.best_model_path is not None
    checkpoint = Path(callback.best_model_path)
    payload = torch.load(checkpoint, weights_only=True)
    manifest = next(tmp_path.glob("*.json"))
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    for state in (payload["callbacks"][0]["state"], metadata["state"]):
        state.pop("final_model_path")
        for option in ("save_every_n_epochs", "save_final", "final_save_path"):
            state["ranking_config"].pop(option)
    torch.save(payload, checkpoint)
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    restored, restored_callback = make_training_battery(tmp_path)
    restored.load_checkpoint(checkpoint)
    assert restored_callback.state_dict() == callback.state_dict()


def test_fresh_callback_does_not_replace_another_runs_final(tmp_path: Path) -> None:
    battery, callback = make_training_battery(tmp_path, save_final=True)
    battery.fit(loader(), loader(), verbose=0)
    assert callback.final_model_path is not None
    original = Path(callback.final_model_path)
    before = original.read_bytes()
    other, other_callback = make_training_battery(tmp_path, save_final=True)
    other.fit(loader(), loader(), verbose=0)
    assert original.read_bytes() == before
    assert other_callback.final_model_path != callback.final_model_path
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_failed_restore_rolls_back_final_path_and_ranking(tmp_path: Path) -> None:
    battery, callback = make_training_battery(tmp_path, save_final=True)
    battery.fit(loader(), loader(), epochs=2, verbose=0)
    restored, restored_callback = make_training_battery(tmp_path, save_final=True)
    before = restored_callback.state_dict()
    files = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert callback.best_model_path is not None
    with (
        patch.object(
            Battery,
            "_restore_global_rng_state",
            side_effect=[RuntimeError("rng"), None],
        ),
        pytest.raises(RuntimeError, match="rng"),
    ):
        restored.load_checkpoint(callback.best_model_path)
    assert restored_callback.state_dict() == before
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == files


@pytest.mark.parametrize("weights_only", [False, True])
def test_failed_first_final_save_keeps_empty_state(
    tmp_path: Path, *, weights_only: bool
) -> None:
    battery, callback = make_training_battery(
        tmp_path,
        save_final=True,
        save_weights_only=weights_only,
        save_every_n_epochs=10,
    )
    before = callback.state_dict()
    with (
        patch.object(callback, "_write_manifest", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        battery.fit(loader(), loader(), verbose=0)
    assert callback.state_dict() == before
    assert not list(tmp_path.iterdir())


def test_legacy_callback_gets_manifest_identity_on_final_save(tmp_path: Path) -> None:
    battery, callback = make_training_battery(tmp_path, save_final=True)
    legacy = callback.state_dict()
    legacy.pop("ranking_id")
    legacy.pop("ranking_config")
    callback.load_state_dict(legacy)
    callback.run_on_train_end({"battery": battery, "model": battery.model, "epoch": 1})
    assert callback.state_dict()["ranking_id"] is not None
    assert_manifest_matches(callback, tmp_path)


def test_final_save_matches_early_stop_results(tmp_path: Path) -> None:
    class StopAfterTwo:
        @charge(Event.AFTER_TRAIN_EPOCH)
        def stop(self, context: EventContext) -> None:
            if context["epoch"] == 2:
                context["battery"].request_stop("test stop")

    battery, callback = make_training_battery(
        tmp_path, save_final=True, extra_callbacks=[StopAfterTwo()]
    )
    result = battery.fit(loader(), loader(), epochs=4, verbose=0)
    assert callback.final_model_path is not None
    payload = torch.load(callback.final_model_path, weights_only=True)
    assert result["stopped_early"] is True
    assert payload["epoch"] == result["epochs_completed"] == 2
    assert payload["results"] == result
    assert payload["stop_reason"] == result["stop_reason"] == "test stop"


def test_named_datasets_produce_one_complete_final(tmp_path: Path) -> None:
    checkpoint = ModelCheckpoint(
        phase="validation", metric="score", save_final=True, save_dir=str(tmp_path)
    )
    model = NamedModel()
    battery = Battery(
        model,
        device="cpu",
        data_pack=SeededNamedPack(),
        callbacks=[checkpoint],
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    )
    result = battery.fit(epochs=2, verbose=0)
    assert checkpoint.final_model_path is not None
    payload = torch.load(checkpoint.final_model_path, weights_only=True)
    assert payload["results"] == result
    assert len(list(tmp_path.glob("final-*"))) == 1
    assert_manifest_matches(checkpoint, tmp_path)


def test_failed_training_does_not_save_final(tmp_path: Path) -> None:
    battery, callback = make_training_battery(tmp_path, save_final=True)
    with (
        patch.object(battery.model, "_step", side_effect=RuntimeError("step failed")),
        pytest.raises(RuntimeError, match="step failed"),
    ):
        battery.fit(loader(), loader(), verbose=0)
    assert callback.final_model_path is None
    assert not list(tmp_path.glob("final-*"))


def test_checkpoint_validation_subscription_keeps_existing_context_and_order(
    tmp_path: Path,
) -> None:
    records: list[tuple[str, dict[str, Any]]] = []

    class Recorder:
        @charge(Event.AFTER_VALIDATION_EPOCH)
        def epoch(self, context: EventContext) -> None:
            records.append(("epoch", dict(context)))
            assert len(list(tmp_path.glob("epoch=*.pth"))) == 1

        @charge(Event.AFTER_VALIDATION)
        def validation(self, context: EventContext) -> None:
            records.append(("validation", dict(context)))

    battery, _ = make_training_battery(tmp_path, extra_callbacks=[Recorder()])
    battery.fit(loader(), loader(), verbose=0)
    assert [event for event, _ in records] == ["epoch", "validation"]
    assert set(records[0][1]) == {
        "battery",
        "model",
        "epoch",
        "val_loss",
        "val_metrics",
    }
    assert "history_val_loss" in records[1][1]


@pytest.mark.parametrize("invalid", [None, 0, "yes"])
def test_final_flag_requires_boolean(tmp_path: Path, invalid: object) -> None:
    with pytest.raises(TypeError, match="save_final"):
        make_training_battery(tmp_path, save_final=invalid)
