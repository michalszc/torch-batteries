"""Tests for public DataPack value contracts."""

from typing import Any

import pytest
import torch
from torch.utils.data import BatchSampler, SequentialSampler, TensorDataset

from torch_batteries import DataContext, DataLoaderConfig, DataPack, DatasetBundle


def test_dataset_bundle_selects_datasets_by_phase() -> None:
    dataset = TensorDataset(torch.arange(4))
    bundle = DatasetBundle(train=dataset, test=dataset)

    assert bundle.for_phase("train") is dataset
    assert bundle.for_phase("validation") is None
    assert bundle.for_phase("test") is dataset


def test_dataset_bundle_normalizes_named_and_singular_datasets() -> None:
    first = TensorDataset(torch.arange(2))
    second = TensorDataset(torch.arange(3))
    named = {"in_domain": first, "out_of_domain": second}
    bundle = DatasetBundle(train=first, test=named, predict=second)

    assert bundle.for_phase("test") is named
    assert bundle.datasets_for_phase("train") == {"default": first}
    assert bundle.datasets_for_phase("test") == named
    assert bundle.datasets_for_phase("predict") == {"default": second}
    assert bundle.datasets_for_phase("validation") == {}


@pytest.mark.parametrize("phase", ["train", "validation"])
def test_dataset_bundle_rejects_named_training_datasets(phase: str) -> None:
    dataset = TensorDataset(torch.arange(2))

    with pytest.raises(TypeError, match=rf"{phase} dataset must be .*Dataset"):
        DatasetBundle(**{phase: {"named": dataset}})  # type: ignore[arg-type]


@pytest.mark.parametrize("phase", ["train", "validation", "test", "predict"])
def test_dataset_bundle_rejects_unsupported_dataset_objects(phase: str) -> None:
    with pytest.raises(TypeError, match=rf"{phase} dataset must be .*got object"):
        DatasetBundle(**{phase: object()})  # type: ignore[arg-type]


@pytest.mark.parametrize("phase", ["test", "predict"])
def test_dataset_bundle_rejects_empty_named_datasets(phase: str) -> None:
    with pytest.raises(ValueError, match=f"{phase} dataset mapping cannot be empty"):
        DatasetBundle(**{phase: {}})  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "   ", 1])
def test_dataset_bundle_rejects_invalid_dataset_names(name: object) -> None:
    dataset = TensorDataset(torch.arange(2))

    with pytest.raises(ValueError, match="names must be non-blank strings"):
        DatasetBundle(test={name: dataset})  # type: ignore[dict-item]


def test_dataset_bundle_rejects_invalid_named_dataset_values() -> None:
    with pytest.raises(TypeError, match="must be a PyTorch Dataset"):
        DatasetBundle(predict={"invalid": object()})  # type: ignore[dict-item]


def test_data_context_supports_public_fields() -> None:
    data_pack = DataPack()
    generator = torch.Generator().manual_seed(7)
    context: DataContext = {
        "data_pack": data_pack,
        "stage": "fit",
        "phase": "train",
        "seed": 7,
        "generator": generator,
        "device": torch.device("cpu"),
    }

    assert context["data_pack"] is data_pack
    assert context["generator"].initial_seed() == 7


def test_data_pack_default_state_contract_is_stateless() -> None:
    data_pack = DataPack()

    assert data_pack.state_dict() == {}
    data_pack.load_state_dict({})


def test_data_pack_default_state_warns_about_unexpected_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_pack = DataPack()

    data_pack.load_state_dict({"split": [1, 2]})

    assert "ignored unexpected state keys: ['split']" in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"num_workers": -1}, "num_workers"),
        ({"num_workers": True}, "num_workers"),
        ({"timeout": -1}, "timeout"),
        ({"prefetch_factor": 0}, "prefetch_factor"),
        ({"persistent_workers": True}, "persistent_workers"),
        ({"prefetch_factor": 2}, "prefetch_factor"),
    ],
)
def test_loader_config_rejects_invalid_values(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DataLoaderConfig(**kwargs)


def test_loader_config_rejects_sampler_with_shuffle() -> None:
    dataset = TensorDataset(torch.arange(4))
    sampler = SequentialSampler(dataset)

    with pytest.raises(ValueError, match="mutually exclusive"):
        DataLoaderConfig(sampler=sampler, shuffle=True)


def test_loader_config_rejects_batch_sampler_conflicts() -> None:
    dataset = TensorDataset(torch.arange(4))
    batch_sampler = BatchSampler(
        SequentialSampler(dataset), batch_size=2, drop_last=False
    )

    with pytest.raises(ValueError, match="batch_sampler requires"):
        DataLoaderConfig(batch_sampler=batch_sampler)

    config = DataLoaderConfig(batch_size=None, batch_sampler=batch_sampler)
    assert config.batch_sampler is batch_sampler
