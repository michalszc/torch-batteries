"""End-to-end tests for Battery workflows backed by a DataPack."""

from contextlib import closing
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import (
    Battery,
    DataContext,
    DataLoaderConfig,
    DataPack,
    DatasetBundle,
    Event,
    EventContext,
    StepOutput,
    charge,
)


class WorkflowModel(nn.Module):
    def __init__(self, *, fail_training: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.fail_training = fail_training
        self.test_dataset_contexts: list[str | None] = []
        self.predict_dataset_contexts: list[str | None] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear.forward(inputs)

    def _step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        predictions = self(inputs)
        return StepOutput(
            loss=F.mse_loss(predictions, targets),
            predictions=predictions,
            targets=targets,
        )

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        if self.fail_training:
            msg = "intentional DataPack workflow failure"
            raise RuntimeError(msg)
        return self._step(context)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self._step(context)

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        self.test_dataset_contexts.append(context.get("dataset_name"))
        return self._step(context)

    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> torch.Tensor:
        self.predict_dataset_contexts.append(context.get("dataset_name"))
        inputs, _ = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return self.forward(inputs)


class WorkflowDataPack(DataPack):
    seed = 11

    def __init__(self, *, include_predict: bool = True) -> None:
        inputs = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]])
        targets = inputs.sum(dim=1, keepdim=True)
        self.dataset = TensorDataset(inputs, targets)
        self.include_predict = include_predict
        self.prepare_calls = 0
        self.setup_stages: list[str] = []
        self.loader_phases: list[str] = []
        self.loader_datasets: list[str | None] = []
        self.generator_seeds: list[int] = []
        self.teardown_stages: list[str] = []

    @charge(Event.PREPARE_DATA)
    def prepare(self, context: DataContext) -> None:
        self.prepare_calls += 1
        self.generator_seeds.append(context["generator"].initial_seed())

    @charge(Event.SETUP_DATA)
    def setup(self, context: DataContext) -> DatasetBundle:
        self.setup_stages.append(context["stage"])
        self.generator_seeds.append(context["generator"].initial_seed())
        return DatasetBundle(
            train=self.dataset,
            validation=self.dataset,
            test=self.dataset,
            predict=self.dataset if self.include_predict else None,
        )

    @charge(Event.CONFIGURE_DATALOADER)
    def configure(self, context: DataContext) -> DataLoaderConfig:
        self.loader_phases.append(context["phase"])
        self.loader_datasets.append(context.get("dataset_name"))
        self.generator_seeds.append(context["generator"].initial_seed())
        return DataLoaderConfig(batch_size=2)

    @charge(Event.TEARDOWN_DATA)
    def teardown(self, context: DataContext) -> None:
        self.teardown_stages.append(context["stage"])


def _battery(data_pack: DataPack, *, fail_training: bool = False) -> Battery:
    model = WorkflowModel(fail_training=fail_training)
    return Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        data_pack=data_pack,
    )


class MultipleWorkflowDataPack(WorkflowDataPack):
    def __init__(self) -> None:
        super().__init__()
        inputs = torch.tensor([[5.0, 1.0], [6.0, 2.0]])
        self.secondary_dataset = TensorDataset(
            inputs,
            inputs.sum(dim=1, keepdim=True),
        )

    @charge(Event.SETUP_DATA)
    def setup(self, context: DataContext) -> DatasetBundle:
        self.setup_stages.append(context["stage"])
        self.generator_seeds.append(context["generator"].initial_seed())
        return DatasetBundle(
            test={"in_domain": self.dataset, "out_of_domain": self.secondary_dataset},
            predict={
                "in_domain": self.dataset,
                "out_of_domain": self.secondary_dataset,
            },
        )


def test_data_pack_drives_all_battery_workflows() -> None:
    data_pack = WorkflowDataPack()
    battery = _battery(data_pack)

    assert battery.data_pack is data_pack

    train_result = battery.train(epochs=1, verbose=0)
    test_result = battery.test(verbose=0)
    predict_result = cast(
        "dict[str, Any]",
        battery.predict(verbose=0, concatenate=True, move_to_cpu=True),
    )
    streamed = list(battery.predict_iter(verbose=0, move_to_cpu=True))

    assert len(train_result["train_loss"]) == 1
    assert len(train_result["val_loss"]) == 1
    assert "test_loss" in test_result
    assert predict_result["predictions"].shape == (4, 1)
    assert len(streamed) == 2
    assert data_pack.prepare_calls == 1
    assert data_pack.setup_stages == ["fit", "test", "predict", "predict"]
    assert data_pack.teardown_stages == ["fit", "test", "predict", "predict"]
    assert data_pack.loader_phases == [
        "train",
        "validation",
        "test",
        "predict",
        "predict",
    ]
    assert data_pack.generator_seeds == [11, 11, 12, 13, 11, 14, 11, 15, 11, 15]


def test_multiple_test_and_prediction_datasets_run_independently() -> None:
    class ResetTrackingMetric:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.samples = 0

        def reset(self) -> None:
            self.reset_calls += 1
            self.samples = 0

        def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
            self.samples += predictions.shape[0]

        def compute(self) -> float:
            return float(self.samples)

    data_pack = MultipleWorkflowDataPack()
    model = WorkflowModel()
    metric = ResetTrackingMetric()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        metrics={"samples": metric},
        data_pack=data_pack,
    )

    raw_test_results = battery.test(verbose=0)
    raw_prediction_results = battery.predict(
        verbose=0,
        concatenate=True,
        move_to_cpu=True,
    )
    test_results = cast("dict[str, Any]", raw_test_results)
    prediction_results = cast("dict[str, Any]", raw_prediction_results)

    assert set(test_results) == {"in_domain", "out_of_domain"}
    assert test_results["in_domain"]["test_metrics"]["samples"] == 4.0
    assert test_results["out_of_domain"]["test_metrics"]["samples"] == 2.0
    assert set(prediction_results) == {"in_domain", "out_of_domain"}
    assert prediction_results["in_domain"]["predictions"].shape == (4, 1)
    assert prediction_results["out_of_domain"]["predictions"].shape == (2, 1)
    assert metric.reset_calls == 2
    assert data_pack.loader_datasets.count("in_domain") == 2
    assert data_pack.loader_datasets.count("out_of_domain") == 2
    assert data_pack.generator_seeds == [11, 11, 14, 14, 11, 15, 15]
    assert model.test_dataset_contexts.count("in_domain") == 2
    assert model.test_dataset_contexts.count("out_of_domain") == 1
    assert model.predict_dataset_contexts.count("in_domain") == 2
    assert model.predict_dataset_contexts.count("out_of_domain") == 1
    assert data_pack.teardown_stages == ["test", "predict"]


def test_named_dataset_selection_returns_singular_result() -> None:
    data_pack = MultipleWorkflowDataPack()
    battery = _battery(data_pack)

    test_result = battery.test(verbose=0, dataset="out_of_domain")
    prediction_result = cast(
        "dict[str, Any]",
        battery.predict(
            verbose=0,
            dataset="out_of_domain",
            concatenate=True,
        ),
    )

    assert "test_loss" in test_result
    assert "out_of_domain" not in test_result
    assert prediction_result["predictions"].shape == (2, 1)
    assert data_pack.loader_datasets == ["out_of_domain", "out_of_domain"]
    assert data_pack.generator_seeds == [11, 11, 14, 11, 15]


def test_named_dataset_errors_are_actionable() -> None:
    data_pack = MultipleWorkflowDataPack()
    battery = _battery(data_pack)
    loader = DataLoader(data_pack.dataset, batch_size=2)

    with pytest.raises(ValueError, match=r"Available datasets.*in_domain"):
        battery.test(verbose=0, dataset="missing")
    with pytest.raises(ValueError, match="explicit test loader"):
        battery.test(  # type: ignore[call-overload]
            loader, verbose=0, dataset="in_domain"
        )
    with pytest.raises(ValueError, match="explicit prediction loader"):
        battery.predict(  # type: ignore[call-overload]
            loader, verbose=0, dataset="in_domain"
        )
    with pytest.raises(ValueError, match="explicit prediction loader"):
        list(battery.predict_iter(loader, verbose=0, dataset="in_domain"))

    assert data_pack.teardown_stages == ["test"]


def test_predict_iter_requires_selection_for_multiple_datasets() -> None:
    data_pack = MultipleWorkflowDataPack()
    battery = _battery(data_pack)

    with pytest.raises(ValueError, match=r"predict_iter\(\) requires dataset="):
        list(battery.predict_iter(verbose=0))
    outputs = list(
        battery.predict_iter(
            verbose=0,
            dataset="out_of_domain",
            move_to_cpu=True,
        )
    )

    assert len(outputs) == 1
    assert outputs[0].device.type == "cpu"
    assert data_pack.loader_datasets == ["out_of_domain"]
    assert data_pack.teardown_stages == ["predict", "predict"]


def test_predict_iter_early_close_tears_down_implicit_data_pack() -> None:
    data_pack = WorkflowDataPack()
    predictions = _battery(data_pack).predict_iter(verbose=0)

    with closing(predictions) as outputs:
        next(outputs)
        assert data_pack.teardown_stages == []

    assert data_pack.teardown_stages == ["predict"]


def test_explicit_loader_uses_direct_mode_without_data_pack_events() -> None:
    data_pack = WorkflowDataPack()
    battery = _battery(data_pack)
    loader = DataLoader(data_pack.dataset, batch_size=2)

    battery.train(loader, epochs=1, verbose=0)
    battery.test(loader, verbose=0)
    battery.predict(loader, verbose=0)

    assert data_pack.prepare_calls == 0
    assert data_pack.setup_stages == []
    assert data_pack.teardown_stages == []


def test_seed_and_generator_are_omitted_when_pack_does_not_declare_seed() -> None:
    contexts: list[DataContext] = []

    class UnseededPack(DataPack):
        @charge(Event.SETUP_DATA)
        def setup(self, context: DataContext) -> DatasetBundle:
            contexts.append(context.copy())
            return DatasetBundle(test=WorkflowDataPack().dataset)

    _battery(UnseededPack()).test(verbose=0)

    assert "seed" not in contexts[0]
    assert "generator" not in contexts[0]


def test_explicit_validation_loader_cannot_mix_with_implicit_training() -> None:
    data_pack = WorkflowDataPack()
    loader = DataLoader(data_pack.dataset, batch_size=2)

    with pytest.raises(ValueError, match="cannot be combined"):
        _battery(data_pack).train(val_loader=loader, verbose=0)


def test_missing_data_pack_produces_actionable_error() -> None:
    model = WorkflowModel()
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    )

    with pytest.raises(ValueError, match="Battery has no DataPack"):
        battery.train(verbose=0)


def test_missing_required_phase_dataset_still_tears_down() -> None:
    data_pack = WorkflowDataPack(include_predict=False)

    with pytest.raises(ValueError, match="phase 'predict'"):
        _battery(data_pack).predict(verbose=0)

    assert data_pack.teardown_stages == ["predict"]


def test_training_failure_still_tears_down() -> None:
    data_pack = WorkflowDataPack()

    with pytest.raises(RuntimeError, match="intentional DataPack workflow failure"):
        _battery(data_pack, fail_training=True).train(verbose=0)

    assert data_pack.teardown_stages == ["fit"]


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_data_pack_seed_is_rejected(seed: object) -> None:
    data_pack = WorkflowDataPack()
    data_pack.seed = seed  # type: ignore[assignment]

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        _battery(data_pack).test(verbose=0)
