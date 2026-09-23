"""Tests for reproducible RNG and DataLoader generator checkpoints."""

import random
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from torch_batteries import (
    Battery,
    DataContext,
    DataLoaderConfig,
    DataPack,
    DatasetBundle,
    Event,
    EventContext,
    StepOutput,
    TrainResult,
    charge,
)


class _DropoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(3, 1)

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        predictions = self.linear(self.dropout(inputs))
        return StepOutput(loss=((predictions - targets) ** 2).mean())


class _SeededDataPack(DataPack):
    seed = 71

    def __init__(self) -> None:
        inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
        targets = inputs.sum(dim=1, keepdim=True)
        self.dataset = TensorDataset(inputs, targets)

    @charge(Event.SETUP_DATA)
    def setup(self, _: DataContext) -> DatasetBundle:
        return DatasetBundle(train=self.dataset)

    @charge(Event.CONFIGURE_DATALOADER)
    def configure(self, _: DataContext) -> DataLoaderConfig:
        return DataLoaderConfig(batch_size=2, shuffle=True)


def _explicit_loader(seed: int = 71) -> DataLoader:
    inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
    targets = inputs.sum(dim=1, keepdim=True)
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _battery(*, with_data_pack: bool = False) -> Battery:
    model = _DropoutModel()
    return Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.2),
        data_pack=_SeededDataPack() if with_data_pack else None,
    )


def _train(
    battery: Battery,
    loader: DataLoader | None,
    epochs: int,
    *,
    resume_from: Path | None = None,
) -> TrainResult:
    return battery.train(
        loader,
        epochs=epochs,
        verbose=0,
        resume_from=resume_from,
        resume_epochs_mode="total",
    )


@pytest.mark.parametrize("loader_mode", ["explicit", "data_pack"])
def test_epoch_boundary_resume_matches_uninterrupted_training(
    tmp_path: Path, loader_mode: str
) -> None:
    with_data_pack = loader_mode == "data_pack"
    torch.manual_seed(19)
    uninterrupted = _battery(with_data_pack=with_data_pack)
    uninterrupted_loader = None if with_data_pack else _explicit_loader()
    uninterrupted_result = _train(uninterrupted, uninterrupted_loader, 4)

    torch.manual_seed(19)
    interrupted = _battery(with_data_pack=with_data_pack)
    interrupted_loader = None if with_data_pack else _explicit_loader()
    _train(interrupted, interrupted_loader, 2)
    checkpoint = tmp_path / "resume.pth"
    interrupted.save_checkpoint(checkpoint)

    torch.manual_seed(999)
    restored = _battery(with_data_pack=with_data_pack)
    restored_loader = None if with_data_pack else _explicit_loader(seed=999)
    restored_result = _train(restored, restored_loader, 4, resume_from=checkpoint)

    assert restored_result == uninterrupted_result
    for actual, expected in zip(
        restored.model.state_dict().values(),
        uninterrupted.model.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(actual, expected)


def test_checkpoint_restores_python_torch_and_numpy_sequences(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    random.seed(11)
    torch.manual_seed(12)
    numpy.random.seed(13)
    source = _battery()
    checkpoint = tmp_path / "rng.pth"
    source.save_checkpoint(checkpoint)
    expected = (random.random(), torch.rand(3), numpy.random.random(3))

    random.seed(91)
    torch.manual_seed(92)
    numpy.random.seed(93)
    restored = _battery()
    restored.load_checkpoint(checkpoint)
    actual = (random.random(), torch.rand(3), numpy.random.random(3))

    assert actual[0] == expected[0]
    assert torch.equal(actual[1], expected[1])
    assert numpy.array_equal(actual[2], expected[2])


def test_distinct_loader_and_sampler_generators_are_captured() -> None:
    dataset = TensorDataset(torch.arange(4))
    loader_generator = torch.Generator().manual_seed(1)
    sampler_generator = torch.Generator().manual_seed(2)
    sampler = RandomSampler(dataset, generator=sampler_generator)
    loader = DataLoader(dataset, sampler=sampler, generator=loader_generator)

    generators = Battery._loader_generators(loader)  # noqa: SLF001

    assert generators == {
        "loader.generator": loader_generator,
        "sampler.generator": sampler_generator,
    }


def test_missing_loader_generator_warns_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = _battery()
    source.train(_explicit_loader(), verbose=0)
    checkpoint = tmp_path / "generator.pth"
    source.save_checkpoint(checkpoint)
    target = _battery()
    loader_without_generator = DataLoader(
        TensorDataset(torch.ones(2, 3), torch.zeros(2, 1)),
        batch_size=1,
    )

    target.train(
        loader_without_generator,
        epochs=2,
        verbose=0,
        resume_from=checkpoint,
    )

    assert "Saved loader generator is unavailable" in caplog.text


def test_incompatible_loader_generator_warns_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    battery = _battery()
    loader = _explicit_loader()
    battery._pending_loader_generator_states = {  # noqa: SLF001
        "train": {"loader.generator": torch.tensor([1])}
    }

    battery._restore_loader_generator_state("train", loader)  # noqa: SLF001

    assert "Saved loader generator is incompatible" in caplog.text


def test_rng_capture_without_numpy_omits_numpy_state() -> None:
    with (
        patch("torch_batteries.trainer._checkpoint.logger.debug") as debug,
        patch(
            "torch_batteries.trainer._checkpoint.importlib.import_module",
            side_effect=ModuleNotFoundError,
        ),
    ):
        state = Battery._capture_global_rng_state()  # noqa: SLF001

    assert "numpy" not in state
    debug.assert_called_once_with(
        "NumPy is unavailable; its RNG state was not checkpointed."
    )


def test_accelerator_rng_capture_includes_all_cuda_and_mps_states() -> None:
    cuda_states = [
        torch.tensor([1], dtype=torch.uint8),
        torch.tensor([2], dtype=torch.uint8),
    ]
    mps_state = torch.tensor([3], dtype=torch.uint8)

    with (
        patch(
            "torch_batteries.trainer._checkpoint.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "torch_batteries.trainer._checkpoint.torch.cuda.get_rng_state_all",
            return_value=cuda_states,
        ),
        patch(
            "torch_batteries.trainer._checkpoint.torch.backends.mps.is_available",
            return_value=True,
        ),
        patch(
            "torch_batteries.trainer._checkpoint.torch.mps.get_rng_state",
            return_value=mps_state,
        ),
    ):
        state = Battery._capture_global_rng_state()  # noqa: SLF001

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(state["cuda"], cuda_states, strict=True)
    )
    assert torch.equal(state["mps"], mps_state)


def test_unavailable_saved_accelerators_warn_and_continue(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = Battery._capture_global_rng_state()  # noqa: SLF001
    state["cuda"] = [torch.tensor([1], dtype=torch.uint8)]
    state["mps"] = torch.tensor([2], dtype=torch.uint8)

    with (
        patch(
            "torch_batteries.trainer._checkpoint.torch.cuda.is_available",
            return_value=False,
        ),
        patch(
            "torch_batteries.trainer._checkpoint.torch.backends.mps.is_available",
            return_value=False,
        ),
    ):
        Battery._restore_global_rng_state(state)  # noqa: SLF001

    assert "CUDA is unavailable" in caplog.text
    assert "MPS is unavailable" in caplog.text


def test_available_saved_accelerators_restore_each_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = Battery._capture_global_rng_state()  # noqa: SLF001
    cuda_state = torch.tensor([1], dtype=torch.uint8)
    mps_state = torch.tensor([2], dtype=torch.uint8)
    state["cuda"] = [cuda_state, torch.tensor([3], dtype=torch.uint8)]
    state["mps"] = mps_state

    with (
        patch(
            "torch_batteries.trainer._checkpoint.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "torch_batteries.trainer._checkpoint.torch.cuda.device_count",
            return_value=1,
        ),
        patch(
            "torch_batteries.trainer._checkpoint.torch.cuda.set_rng_state"
        ) as set_cuda,
        patch(
            "torch_batteries.trainer._checkpoint.torch.backends.mps.is_available",
            return_value=True,
        ),
        patch("torch_batteries.trainer._checkpoint.torch.mps.set_rng_state") as set_mps,
    ):
        Battery._restore_global_rng_state(state)  # noqa: SLF001

    set_cuda.assert_called_once()
    set_mps.assert_called_once()
    assert torch.equal(set_cuda.call_args.args[0], cuda_state)
    assert torch.equal(set_mps.call_args.args[0], mps_state)
    assert "CUDA RNG device count differs" in caplog.text


def test_unavailable_numpy_warns_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = Battery._capture_global_rng_state()  # noqa: SLF001

    with patch(
        "torch_batteries.trainer._checkpoint.importlib.import_module",
        side_effect=ModuleNotFoundError,
    ):
        Battery._restore_global_rng_state(state)  # noqa: SLF001

    assert "NumPy is unavailable" in caplog.text


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ([], "Invalid RNG state"),
        ({"python": random.getstate()}, "Invalid RNG state"),
        (
            {"python": (0, (), None), "torch_cpu": torch.get_rng_state()},
            "Invalid Python RNG state",
        ),
        (
            {
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
                "cuda": {},
            },
            "Invalid CUDA RNG state",
        ),
        (
            {
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
                "mps": [],
            },
            "Invalid MPS RNG state",
        ),
        (
            {
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
                "numpy": {},
            },
            "Invalid NumPy RNG state",
        ),
    ],
)
def test_invalid_global_rng_state_is_rejected(state: object, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        Battery._validate_global_rng_state(state)  # noqa: SLF001


@pytest.mark.parametrize(
    "state",
    [[], {1: {}}, {"train": []}, {"train": {1: torch.get_rng_state()}}],
)
def test_invalid_loader_generator_state_is_rejected(state: object) -> None:
    with pytest.raises(TypeError, match="Invalid loader generator state"):
        Battery._validate_loader_generator_states(state)  # noqa: SLF001
