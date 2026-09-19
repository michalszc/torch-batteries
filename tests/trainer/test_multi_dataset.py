"""Named dataset scheduling and weighted result contracts."""

from typing import Any, Literal

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from torch_batteries import (
    BatchScheduleConfig,
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
from torch_batteries.callbacks import EarlyStopping


class NamedPack(DataPack):
    def __init__(
        self, train_mode: Literal["round_robin", "interleave"] = "round_robin"
    ) -> None:
        self.train_mode = train_mode
        self.datasets = {
            "small": TensorDataset(torch.ones(5, 1)),
            "large": TensorDataset(torch.full((6, 1), 3.0)),
        }

    @charge(Event.SETUP_DATA)
    def setup(self, _: DataContext) -> DatasetBundle:
        return DatasetBundle(
            train=self.datasets,
            validation=self.datasets,
            test=self.datasets,
            predict=self.datasets,
            train_batch_schedule=BatchScheduleConfig(self.train_mode, seed=7),
            validation_batch_schedule=BatchScheduleConfig("interleave", seed=11),
        )

    @charge(Event.CONFIGURE_DATALOADER)
    def configure(self, context: DataContext) -> DataLoaderConfig:
        return DataLoaderConfig(
            batch_size=2 if context["dataset_name"] == "small" else 3,
            shuffle=False,
        )


class SeededNamedPack(NamedPack):
    def __init__(self) -> None:
        super().__init__()
        self.generators: dict[tuple[str, str], torch.Generator] = {}

    @charge(Event.CONFIGURE_DATALOADER)
    def configure(self, context: DataContext) -> DataLoaderConfig:
        key = (context["phase"], context["dataset_name"])
        generator = torch.Generator().manual_seed(13 if key[1] == "small" else 17)
        self.generators[key] = generator
        return DataLoaderConfig(batch_size=2, shuffle=True, generator=generator)


class NamedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.train_order: list[str] = []
        self.validation_order: list[str] = []

    def _step(self, context: EventContext, phase: str) -> StepOutput:
        name = context["dataset_name"]
        if phase == "train":
            self.train_order.append(name)
        elif phase == "validation":
            self.validation_order.append(name)
        value = 1.0 if name == "small" else 3.0
        return StepOutput(
            loss=self.bias * 0 + value,
            metrics={"score": 0.2 if name == "small" else 0.8},
        )

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        return self._step(context, "train")

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self._step(context, "validation")

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        return self._step(context, "test")

    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> Any:
        return context["batch"][0]


class AutomaticNamedModel(NamedModel):
    def _step(self, context: EventContext, phase: str) -> StepOutput:
        output = super()._step(context, phase)
        values = context["batch"][0]
        output.predictions = values
        output.targets = torch.zeros_like(values)
        return output


class SampleCountMetric:
    def __init__(self) -> None:
        self.samples = 0

    def reset(self) -> None:
        self.samples = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self.samples += predictions.shape[0]

    def compute(self) -> float:
        return float(self.samples)

    def state_dict(self) -> dict[str, int]:
        return {"samples": self.samples}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.samples = state["samples"]


def _battery(
    mode: Literal["round_robin", "interleave"] = "round_robin",
) -> tuple[Battery, NamedModel]:
    model = NamedModel()
    return (
        Battery(
            model,
            device="cpu",
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            data_pack=NamedPack(mode),
        ),
        model,
    )


def test_round_robin_exhausts_loaders_and_weights_samples() -> None:
    battery, model = _battery()
    result = battery.fit(epochs=1, verbose=0)

    assert model.train_order == ["small", "large", "small", "large", "small"]
    assert result["train_loss"] == pytest.approx([23 / 11])
    assert result["val_loss"] == pytest.approx([23 / 11])
    assert result["train_metrics"]["score"] == pytest.approx([5.8 / 11])
    assert result["train_metrics"]["small:loss"] == pytest.approx([1.0])
    assert result["train_metrics"]["large:loss"] == pytest.approx([3.0])
    assert result["val_metrics"]["small:score"] == pytest.approx([0.2])
    assert result["val_metrics"]["large:score"] == pytest.approx([0.8])


def test_seeded_interleave_is_reproducible_and_exhaustive() -> None:
    first, first_model = _battery("interleave")
    second, second_model = _battery("interleave")
    first.train(epochs=1, verbose=0)
    second.train(epochs=1, verbose=0)

    assert first_model.train_order == second_model.train_order
    assert first_model.train_order.count("small") == 3
    assert first_model.train_order.count("large") == 2


def test_callbacks_monitor_exact_aggregate_or_dataset_metric_key() -> None:
    for key, expected in (("loss", 23 / 11), ("small:loss", 1.0)):
        model = NamedModel()
        stopping = EarlyStopping(phase="train", metric=key, patience=2)
        battery = Battery(
            model,
            device="cpu",
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            callbacks=[stopping],
            data_pack=NamedPack(),
        )
        battery.train(verbose=0)
        assert stopping.best_score == pytest.approx(expected)


def test_single_named_dataset_uses_unprefixed_metrics() -> None:
    class SingleNamedPack(NamedPack):
        @charge(Event.SETUP_DATA)
        def setup(self, _: DataContext) -> DatasetBundle:
            datasets = {"small": self.datasets["small"]}
            return DatasetBundle(
                train=datasets,
                validation=datasets,
                test=datasets,
                predict=datasets,
            )

    model = NamedModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        data_pack=SingleNamedPack(),
    )
    history = battery.fit(verbose=0)
    validation = battery.validate(verbose=0)
    test = battery.test(verbose=0)
    predictions = battery.predict(verbose=0, concatenate=True)

    assert history["train_metrics"] == {"score": [0.2]}
    assert history["val_metrics"] == {"score": [0.2]}
    assert validation["val_metrics"] == {"score": 0.2}
    assert test["test_metrics"] == {"score": 0.2}
    assert predictions["predictions"].shape == (5, 1)


def test_test_and_prediction_use_flat_results_and_selection() -> None:
    battery, _ = _battery()
    result = battery.test(verbose=0)
    assert result["test_loss"] == pytest.approx(23 / 11)
    assert result["test_metrics"]["score"] == pytest.approx(5.8 / 11)
    assert result["test_metrics"]["small:loss"] == pytest.approx(1.0)
    assert result["test_metrics"]["large:score"] == pytest.approx(0.8)

    predictions = battery.predict(verbose=0, concatenate=True)["predictions"]
    assert set(predictions) == {"small", "large"}
    assert predictions["small"].shape == (5, 1)
    assert predictions["large"].shape == (6, 1)

    selected = battery.test(verbose=0, dataset="small")
    assert selected["test_loss"] == pytest.approx(1.0)
    assert "small:loss" not in selected.get("test_metrics", {})
    selected_predictions = battery.predict(
        verbose=0, dataset="small", concatenate=True
    )["predictions"]
    assert selected_predictions.shape == (5, 1)


def test_stateful_metrics_have_aggregate_and_isolated_dataset_state(
    tmp_path: Any,
) -> None:
    model = AutomaticNamedModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        data_pack=NamedPack(),
        metrics={
            "train": {"samples": SampleCountMetric()},
            "validation": {"samples": SampleCountMetric()},
            "test": {"samples": SampleCountMetric()},
        },
    )

    history = battery.fit(verbose=0)
    test = battery.test(verbose=0)
    for metrics in (history["train_metrics"], history["val_metrics"]):
        assert metrics["samples"] == [11.0]
        assert metrics["small:samples"] == [5.0]
        assert metrics["large:samples"] == [6.0]
    assert test["test_metrics"]["samples"] == 11.0
    assert test["test_metrics"]["small:samples"] == 5.0
    assert test["test_metrics"]["large:samples"] == 6.0

    checkpoint = tmp_path / "metrics.pth"
    battery.save_checkpoint(checkpoint)
    metric_states = torch.load(checkpoint, weights_only=True)["metrics"]
    assert set(metric_states["phases"]) == {"train", "validation", "test"}
    assert set(metric_states["datasets"]["train"]) == {"small", "large"}
    assert metric_states["phases"]["test"]["samples"] == {"samples": 11}

    restored_model = AutomaticNamedModel()
    restored = Battery(
        restored_model,
        device="cpu",
        optimizer=torch.optim.SGD(restored_model.parameters(), lr=0.01),
        data_pack=NamedPack(),
        metrics={
            "train": {"samples": SampleCountMetric()},
            "validation": {"samples": SampleCountMetric()},
            "test": {"samples": SampleCountMetric()},
        },
    )
    restored.load_checkpoint(checkpoint)
    assert restored._checkpoint_metric_states() == metric_states  # noqa: SLF001


def test_named_loader_generator_states_resume_per_dataset(tmp_path: Any) -> None:
    def build(pack: SeededNamedPack) -> Battery:
        model = NamedModel()
        return Battery(
            model,
            device="cpu",
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            data_pack=pack,
        )

    uninterrupted_pack = SeededNamedPack()
    uninterrupted = build(uninterrupted_pack)
    uninterrupted.fit(epochs=2, verbose=0)

    first = build(SeededNamedPack())
    first.fit(epochs=1, verbose=0)
    checkpoint = tmp_path / "named.pth"
    first.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert set(payload["loader_generator_states"]) == {
        "train:small",
        "train:large",
        "validation:small",
        "validation:large",
    }

    resumed_pack = SeededNamedPack()
    resumed = build(resumed_pack)
    result = resumed.fit(epochs=2, resume_from=checkpoint, verbose=0)
    assert len(result["train_loss"]) == 2
    for key, generator in uninterrupted_pack.generators.items():
        assert torch.equal(
            generator.get_state(), resumed_pack.generators[key].get_state()
        )
