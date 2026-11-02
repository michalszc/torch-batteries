"""End-to-end checks for authoritative phase metrics in progress output."""

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.trainer.test_multi_dataset import (
    AutomaticNamedModel,
    NamedPack,
    SampleCountMetric,
)
from tests.utils.test_stateful_metrics import (
    _ControlledMetric,
    _MetricModel,
    _PhaseMean,
)
from torch_batteries import (
    Battery,
    CollectedMetric,
    Event,
    EventContext,
    StepOutput,
    charge,
)
from torch_batteries.utils.progress import (
    BarProgress,
    Phase,
    SilentProgress,
    SimpleProgress,
)

if TYPE_CHECKING:
    from torch_batteries.utils.metrics.types import MetricDefinition


class EvaluationModel(_MetricModel):
    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        return self._step(context)


class ManualMetricModel(EvaluationModel):
    def _step(self, context: EventContext) -> StepOutput:
        output = super()._step(context)
        output.metrics = {"running_mean": 42.0}
        return output


@pytest.mark.parametrize("tracker", [BarProgress, SimpleProgress, SilentProgress])
def test_final_values_override_averages_without_batch_updates(
    tracker: type[BarProgress | SimpleProgress | SilentProgress],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("torch_batteries.utils.progress.progress_bar.tqdm") as tqdm:
        progress = tracker()
        progress.start_epoch(1)
        progress.start_phase(Phase.TRAIN, total_batches=2)
        progress.update({"loss": 1.0}, 2)
        progress.update({"loss": 3.0}, 1)
        values = progress.end_phase(final_metrics={"loss": 0.5, "macro_f1": 0.73})
        progress.end_epoch()
    assert values == {"loss": 0.5, "macro_f1": 0.73}
    if tracker is BarProgress:
        bar = tqdm.return_value
        assert bar.update.call_count == 2
        assert bar.method_calls[-2:] == [
            ("set_postfix_str", ("Loss=0.5000, Macro_f1=0.7300",), {}),
            ("close", (), {}),
        ]
    elif tracker is SimpleProgress:
        assert "Macro F1: 0.7300" in capsys.readouterr().out
    else:
        assert capsys.readouterr().out == ""


@pytest.mark.parametrize("workflow", ["train", "fit", "validate", "test"])
def test_phase_metrics_render_once_before_bar_closes(workflow: str) -> None:
    model = EvaluationModel()
    metric = _PhaseMean()
    collected = MagicMock(side_effect=lambda predictions, targets: predictions.max())
    metrics: dict[str, MetricDefinition] = {
        "batch_mean": lambda predictions, targets: predictions.mean(),
        "running_mean": metric,
        "collected_max": CollectedMetric(collected),
    }
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        metrics=metrics,
    )
    loader = DataLoader(
        TensorDataset(torch.tensor([[1.0], [2.0], [9.0]]), torch.zeros(3, 1)),
        batch_size=2,
    )
    bars: list[MagicMock] = []

    def new_bar(**kwargs: Any) -> MagicMock:
        bar = MagicMock()
        bars.append(bar)
        return bar

    result: Any
    with (
        patch("torch_batteries.utils.progress.progress_bar.tqdm", side_effect=new_bar),
        patch.object(metric, "compute", wraps=metric.compute) as compute,
    ):
        if workflow == "fit":
            result = battery.fit(loader, loader, verbose=1)
        else:
            result = getattr(battery, workflow)(loader, verbose=1)
    phases = 2 if workflow == "fit" else 1
    assert compute.call_count == collected.call_count == len(bars) == phases
    phase = (
        "train"
        if workflow in {"train", "fit"}
        else "val"
        if workflow == "validate"
        else "test"
    )
    values = result[f"{phase}_metrics"]
    assert values["running_mean"] == ([4.0] if phase == "train" else 4.0)
    for bar in bars:
        postfixes = [call.args[0] for call in bar.set_postfix_str.call_args_list]
        assert "Running_mean" not in postfixes[0]
        assert "Running_mean=4.0000" in postfixes[-1]
        assert "Collected_max=9.0000" in postfixes[-1]
        assert bar.method_calls[-1][0] == "close"
        assert bar.update.call_count == 2


@pytest.mark.parametrize("workflow", ["train", "validate", "test"])
@pytest.mark.parametrize("policy", ["raise", "warn"])
def test_phase_compute_failures_close_active_bar(workflow: str, policy: str) -> None:
    model = EvaluationModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        metrics={"broken": _ControlledMetric("compute")},
        metric_error_policy=policy,  # type: ignore[arg-type]
    )
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=1
    )
    with patch("torch_batteries.utils.progress.progress_bar.tqdm") as tqdm:
        if policy == "raise":
            with pytest.raises(RuntimeError):
                getattr(battery, workflow)(loader, verbose=1)
        else:
            result = getattr(battery, workflow)(loader, verbose=1)
            assert "broken" not in str(result)
        tqdm.return_value.close.assert_called_once()


@pytest.mark.parametrize("workflow", ["fit", "validate", "test"])
def test_named_phase_metrics_include_exact_aggregate_and_dataset_values(
    workflow: str,
) -> None:
    model = AutomaticNamedModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        data_pack=NamedPack(),
        metrics={"samples": SampleCountMetric()},
    )
    computed_samples: list[int] = []

    def compute(metric: SampleCountMetric) -> float:
        computed_samples.append(metric.samples)
        return float(metric.samples)

    with (
        patch("torch_batteries.utils.progress.progress_bar.tqdm") as tqdm,
        patch.object(SampleCountMetric, "compute", compute),
    ):
        getattr(battery, workflow)(verbose=1)
    postfix = tqdm.return_value.set_postfix_str.call_args.args[0]
    assert "Samples=11.0000" in postfix
    assert "small:samples=5.0000" in postfix
    assert "large:samples=6.0000" in postfix
    assert sorted(computed_samples) == sorted(
        [5, 6, 11] * (2 if workflow == "fit" else 1)
    )


@pytest.mark.parametrize("workflow", ["train", "validate", "test"])
def test_final_display_preserves_manual_metric_precedence(workflow: str) -> None:
    model = ManualMetricModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        metrics={"running_mean": _PhaseMean()},
    )
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=1
    )
    with patch("torch_batteries.utils.progress.progress_bar.tqdm") as tqdm:
        result = getattr(battery, workflow)(loader, verbose=1)
    assert "Running_mean=42.0000" in tqdm.return_value.set_postfix_str.call_args.args[0]
    assert "42.0" in str(result)


def test_named_aggregate_compute_failure_closes_shared_bar() -> None:
    model = AutomaticNamedModel()
    battery = Battery(
        model,
        device="cpu",
        data_pack=NamedPack(),
        metrics={"samples": SampleCountMetric()},
    )

    def compute(metric: SampleCountMetric) -> float:
        if metric.samples == 11:
            message = "aggregate failure"
            raise RuntimeError(message)
        return float(metric.samples)

    with (
        patch("torch_batteries.utils.progress.progress_bar.tqdm") as tqdm,
        patch.object(SampleCountMetric, "compute", compute),
        pytest.raises(RuntimeError, match="aggregate failure"),
    ):
        battery.test(verbose=1)
    tqdm.return_value.close.assert_called_once()
