"""Tests for resumable Battery checkpoints."""

from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, charge
from torch_batteries.callbacks import (
    GradientAccumulation,
    LearningRateScheduler,
    ModelCheckpoint,
)


class _Model(nn.Module):
    def __init__(self, outputs: int = 1) -> None:
        super().__init__()
        self.layer = nn.Linear(1, outputs)

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> torch.Tensor:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return cast("torch.Tensor", ((self.layer(inputs) - targets) ** 2).mean())


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(4, 1), torch.zeros(4, 1)),
        batch_size=1,
    )


def _battery() -> tuple[Battery, LearningRateScheduler]:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = LearningRateScheduler(StepLR(optimizer, 1, gamma=0.5))
    return (
        Battery(
            model,
            device="cpu",
            optimizer=optimizer,
            callbacks=[GradientAccumulation(2), scheduler],
        ),
        scheduler,
    )


def test_full_checkpoint_resumes_total_epochs(tmp_path: Path) -> None:
    battery, _ = _battery()
    initial = battery.train(_loader(), epochs=2, verbose=0)
    checkpoint = tmp_path / "training.pth"
    battery.save_checkpoint(checkpoint)
    assert torch.load(checkpoint, weights_only=True)["epoch"] == 2
    restored, restored_scheduler = _battery()

    result = restored.train(
        _loader(),
        epochs=4,
        verbose=0,
        resume_from=checkpoint,
        resume_epochs_mode="total",
    )

    assert len(initial["train_loss"]) == 2
    assert len(result["train_loss"]) == 4
    assert restored_scheduler.scheduler.last_epoch == 4


def test_full_checkpoint_resumes_additional_epochs(tmp_path: Path) -> None:
    battery, _ = _battery()
    battery.train(_loader(), epochs=2, verbose=0)
    checkpoint = tmp_path / "training.pth"
    battery.save_checkpoint(checkpoint)
    restored, _ = _battery()

    result = restored.train(
        _loader(),
        epochs=2,
        verbose=0,
        resume_from=checkpoint,
        resume_epochs_mode="additional",
    )

    assert len(result["train_loss"]) == 4


def test_raw_model_state_is_detected_and_warned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = _Model()
    path = tmp_path / "weights.pth"
    torch.save(source.state_dict(), path)
    target = _Model()
    battery = Battery(target, device="cpu")

    battery.load_checkpoint(path)

    assert "Raw model state detected" in caplog.text
    for source_value, target_value in zip(
        source.state_dict().values(), target.state_dict().values(), strict=True
    ):
        assert torch.equal(source_value, target_value)


def test_raw_model_state_load_is_always_strict(tmp_path: Path) -> None:
    path = tmp_path / "weights.pth"
    torch.save(_Model(outputs=2).state_dict(), path)
    battery = Battery(_Model(outputs=1), device="cpu")

    with pytest.raises(RuntimeError):
        battery.load_checkpoint(path)


def test_rejects_malformed_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pth"
    torch.save({"unexpected": "value"}, path)
    battery = Battery(_Model(), device="cpu")

    with pytest.raises(ValueError, match="Unrecognized"):
        battery.load_checkpoint(path)


def test_rejects_unsupported_full_checkpoint_schema(tmp_path: Path) -> None:
    source, _ = _battery()
    path = tmp_path / "unsupported-schema.pth"
    source.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    payload["__torch_batteries_checkpoint__"] = 2
    torch.save(payload, path)
    target, _ = _battery()

    with pytest.raises(ValueError, match="schema 2 is unsupported"):
        target.load_checkpoint(path)


def test_rejects_full_checkpoint_with_missing_fields(tmp_path: Path) -> None:
    source, _ = _battery()
    path = tmp_path / "source.pth"
    source.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    del payload["model"]
    malformed = tmp_path / "missing.pth"
    torch.save(payload, malformed)
    target, _ = _battery()

    with pytest.raises(ValueError, match="missing fields"):
        target.load_checkpoint(malformed)


def test_resume_requires_optimizer_when_checkpoint_contains_one(
    tmp_path: Path,
) -> None:
    source, _ = _battery()
    path = tmp_path / "source.pth"
    source.save_checkpoint(path)
    target = Battery(_Model(), device="cpu")

    with pytest.raises(ValueError, match="optimizer is required"):
        target.load_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "replacement", "exception", "message"),
    [
        ("callbacks", {}, TypeError, "Invalid callback state"),
        ("metrics", [], TypeError, "Invalid metric state"),
        ("results", [], TypeError, "Invalid training history"),
    ],
)
def test_rejects_invalid_full_checkpoint_field_types(
    tmp_path: Path,
    field: str,
    replacement: object,
    exception: type[Exception],
    message: str,
) -> None:
    source, _ = _battery()
    path = tmp_path / f"{field}.pth"
    source.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    payload[field] = replacement
    torch.save(payload, path)
    target, _ = _battery()

    with pytest.raises(exception, match=message):
        target.load_checkpoint(path)


def test_rejects_callback_order_mismatch(tmp_path: Path) -> None:
    battery, _ = _battery()
    battery.train(_loader(), verbose=0)
    path = tmp_path / "training.pth"
    battery.save_checkpoint(path)
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    reordered = Battery(
        model,
        device="cpu",
        optimizer=optimizer,
        callbacks=[
            LearningRateScheduler(StepLR(optimizer, 1, gamma=0.5)),
            GradientAccumulation(2),
        ],
    )

    with pytest.raises(ValueError, match="callbacks do not match"):
        reordered.load_checkpoint(path)


def test_rejects_invalid_resume_mode_and_completed_total_target(
    tmp_path: Path,
) -> None:
    battery, _ = _battery()
    with pytest.raises(ValueError, match="resume_epochs_mode"):
        battery.train(
            _loader(),
            verbose=0,
            resume_epochs_mode="invalid",
        )

    battery.train(_loader(), epochs=1, verbose=0)
    path = tmp_path / "complete.pth"
    battery.save_checkpoint(path)
    restored, _ = _battery()
    with pytest.raises(ValueError, match="does not contain any new epochs"):
        restored.train(
            _loader(),
            epochs=1,
            verbose=0,
            resume_from=path,
            resume_epochs_mode="total",
        )


def test_model_checkpoint_saves_full_training_state_by_default(
    tmp_path: Path,
) -> None:
    model = _Model()
    callback = ModelCheckpoint(
        stage="train",
        metric="loss",
        mode="min",
        save_dir=str(tmp_path),
        save_path="full.pth",
    )
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[callback],
    )

    battery.train(_loader(), verbose=0)

    payload = torch.load(tmp_path / "full.pth", weights_only=True)
    assert payload["__torch_batteries_checkpoint__"] == 1


def test_model_checkpoint_can_save_raw_weights(tmp_path: Path) -> None:
    model = _Model()
    callback = ModelCheckpoint(
        stage="train",
        metric="loss",
        mode="min",
        save_dir=str(tmp_path),
        save_path="weights.pth",
        save_weights_only=True,
    )
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[callback],
    )

    battery.train(_loader(), verbose=0)

    payload = torch.load(tmp_path / "weights.pth", weights_only=True)
    assert set(payload) == set(model.state_dict())
