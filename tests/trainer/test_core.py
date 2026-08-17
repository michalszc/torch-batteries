"""Tests for torch_batteries.trainer module."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from torch_batteries.events import Event, EventContext, charge
from torch_batteries.trainer import Battery, StepOutput
from torch_batteries.utils.progress import SilentProgress


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # type: ignore[no-any-return]

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        batch = context["batch"]
        x, y = batch
        pred = self(x)
        return StepOutput(
            loss=nn.functional.mse_loss(pred, y), predictions=pred, targets=y
        )

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        batch = context["batch"]
        x, y = batch
        pred = self(x)
        return StepOutput(
            loss=nn.functional.mse_loss(pred, y), predictions=pred, targets=y
        )

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        batch = context["batch"]
        x, y = batch
        pred = self(x)
        return StepOutput(
            loss=nn.functional.mse_loss(pred, y), predictions=pred, targets=y
        )

    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> torch.Tensor:
        batch = context["batch"]
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x)  # type: ignore[no-any-return]


class ContextRecorder:
    """Record selected event contexts for trainer tests."""

    def __init__(self) -> None:
        self.after_train: list[EventContext] = []
        self.before_validation: list[EventContext] = []
        self.after_validation: list[EventContext] = []
        self.after_train_step: list[EventContext] = []
        self.after_validation_step: list[EventContext] = []
        self.after_test_step: list[EventContext] = []
        self.after_test: list[EventContext] = []

    @charge(Event.AFTER_TRAIN)
    def record_after_train(self, context: EventContext) -> None:
        self.after_train.append(context.copy())

    @charge(Event.BEFORE_VALIDATION)
    def record_before_validation(self, context: EventContext) -> None:
        self.before_validation.append(context.copy())

    @charge(Event.AFTER_VALIDATION)
    def record_after_validation(self, context: EventContext) -> None:
        self.after_validation.append(context.copy())

    @charge(Event.AFTER_TRAIN_STEP)
    def record_after_train_step(self, context: EventContext) -> None:
        self.after_train_step.append(context.copy())

    @charge(Event.AFTER_VALIDATION_STEP)
    def record_after_validation_step(self, context: EventContext) -> None:
        self.after_validation_step.append(context.copy())

    @charge(Event.AFTER_TEST_STEP)
    def record_after_test_step(self, context: EventContext) -> None:
        self.after_test_step.append(context.copy())

    @charge(Event.AFTER_TEST)
    def record_after_test(self, context: EventContext) -> None:
        self.after_test.append(context.copy())


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error metric used in trainer context tests."""
    return torch.mean(torch.abs(pred - target))


class TestBattery:
    """Test cases for Battery trainer class."""

    def create_simple_data_loader(
        self, batch_size: int = 4, num_samples: int = 16, input_size: int = 10
    ) -> DataLoader:
        """Create a simple data loader for testing."""
        x = torch.randn(num_samples, input_size)
        y = torch.randn(num_samples, 1)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size)

    def test_init_with_defaults(self) -> None:
        """Test Battery initialization with default values."""
        model = SimpleModel()
        battery = Battery(model)

        assert battery.model is model
        assert isinstance(battery.device, torch.device)
        assert battery.optimizer is None

    def test_init_with_custom_device(self) -> None:
        """Test Battery initialization with custom device."""
        model = SimpleModel()
        battery = Battery(model, device="cpu")

        assert battery.device.type == "cpu"

    def test_init_with_optimizer(self) -> None:
        """Test Battery initialization with optimizer."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer)

        assert battery.optimizer is optimizer

    def test_optimizer_property_setter(self) -> None:
        """Test optimizer property setter."""
        model = SimpleModel()
        battery = Battery(model)
        optimizer = optim.Adam(model.parameters())

        battery.optimizer = optimizer
        assert battery.optimizer is optimizer

    @patch("torch_batteries.trainer.core.get_device")
    def test_auto_device_detection(self, mock_get_device: MagicMock) -> None:
        """Test auto device detection."""
        mock_device = torch.device("cpu")
        mock_get_device.return_value = mock_device

        model = SimpleModel()
        battery = Battery(model, device="auto")

        mock_get_device.assert_called_once_with("auto")
        assert battery.device == mock_device

    def test_model_moved_to_device(self) -> None:
        """Test that model is moved to specified device."""
        model = SimpleModel()
        battery = Battery(model, device="cpu")

        # Check that model parameters are on correct device
        for param in battery.model.parameters():
            assert param.device.type == "cpu"

    def test_train_without_optimizer_raises_error(self) -> None:
        """Test that training without optimizer raises ValueError."""
        model = SimpleModel()
        battery = Battery(model)
        train_loader = self.create_simple_data_loader()

        with pytest.raises(ValueError, match="Optimizer is required for training"):
            battery.train(train_loader)

    def test_train_rejects_non_positive_epochs(self) -> None:
        """Training requires at least one epoch."""
        model = SimpleModel()
        battery = Battery(model, optimizer=optim.SGD(model.parameters(), lr=0.01))

        with pytest.raises(ValueError, match="epochs must be greater than zero"):
            battery.train(self.create_simple_data_loader(), epochs=0)

    def test_workflows_reject_empty_loaders(self) -> None:
        """Train, validation, test, and prediction reject empty loaders."""
        model = SimpleModel()
        battery = Battery(model, optimizer=optim.SGD(model.parameters(), lr=0.01))
        empty_loader = DataLoader(
            TensorDataset(torch.empty(0, 10), torch.empty(0, 1)), batch_size=2
        )
        loader = self.create_simple_data_loader()

        with pytest.raises(ValueError, match="Training loader must not be empty"):
            battery.train(empty_loader)
        with pytest.raises(ValueError, match="Validation loader must not be empty"):
            battery.train(loader, empty_loader)
        with pytest.raises(ValueError, match="Test loader must not be empty"):
            battery.test(empty_loader)
        with pytest.raises(ValueError, match="Prediction loader must not be empty"):
            battery.predict(empty_loader)

    def test_train_resets_stop_training_flag(self) -> None:
        """A Battery remains reusable after a previous stop request."""
        model = SimpleModel()
        battery = Battery(model, optimizer=optim.SGD(model.parameters(), lr=0.01))
        battery.stop_training = True

        result = battery.train(self.create_simple_data_loader(), verbose=0)

        assert len(result["train_loss"]) == 1
        assert battery.stop_training is False

    def test_training_error_aborts_progress_and_propagates(self) -> None:
        """Training releases progress resources without hiding step errors."""

        class FailingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.parameter = nn.Parameter(torch.tensor(1.0))

            @charge(Event.TRAIN_STEP)
            def training_step(self, _: EventContext) -> torch.Tensor:
                msg = "step failed"
                raise RuntimeError(msg)

        progress = SilentProgress()
        model = FailingModel()
        battery = Battery(model, optimizer=optim.SGD(model.parameters(), lr=0.01))

        with (
            patch(
                "torch_batteries.trainer._training.ProgressFactory.create",
                return_value=progress,
            ),
            patch.object(progress, "abort", wraps=progress.abort) as abort,
            pytest.raises(RuntimeError, match="step failed"),
        ):
            battery.train(self.create_simple_data_loader(), verbose=0)

        abort.assert_called_once_with()

    @pytest.mark.parametrize("workflow", ["validation", "test", "predict"])
    def test_phase_callback_errors_abort_progress_and_propagate(
        self, workflow: str
    ) -> None:
        """Interrupted evaluation phases abort without success lifecycle events."""

        class FailingPhaseCallback:
            def __init__(self, failing_workflow: str) -> None:
                self.failing_workflow = failing_workflow
                self.completed: set[str] = set()

            def _fail_for(self, phase: str) -> None:
                if self.failing_workflow == phase:
                    msg = f"{phase} callback failed"
                    raise RuntimeError(msg)

            @charge(Event.AFTER_VALIDATION_STEP)
            def fail_validation(self, _: EventContext) -> None:
                self._fail_for("validation")

            @charge(Event.AFTER_TEST_STEP)
            def fail_test(self, _: EventContext) -> None:
                self._fail_for("test")

            @charge(Event.AFTER_PREDICT_STEP)
            def fail_predict(self, _: EventContext) -> None:
                self._fail_for("predict")

            @charge(Event.AFTER_VALIDATION)
            def complete_validation(self, _: EventContext) -> None:
                self.completed.add("validation")

            @charge(Event.AFTER_TEST)
            def complete_test(self, _: EventContext) -> None:
                self.completed.add("test")

            @charge(Event.AFTER_PREDICT)
            def complete_predict(self, _: EventContext) -> None:
                self.completed.add("predict")

            @charge(Event.AFTER_TRAIN)
            def complete_train(self, _: EventContext) -> None:
                self.completed.add("train")

        callback = FailingPhaseCallback(workflow)
        model = SimpleModel()
        battery = Battery(
            model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            callbacks=[callback],
        )
        loader = self.create_simple_data_loader(batch_size=2, num_samples=4)
        progress = SilentProgress()

        def run_workflow() -> None:
            if workflow == "validation":
                battery.train(loader, loader, verbose=0)
            elif workflow == "test":
                battery.test(loader, verbose=0)
            else:
                battery.predict(loader, verbose=0)

        progress_module = {
            "validation": "_training",
            "test": "_evaluation",
            "predict": "_prediction",
        }[workflow]

        with (
            patch(
                f"torch_batteries.trainer.{progress_module}.ProgressFactory.create",
                return_value=progress,
            ),
            patch.object(progress, "abort", wraps=progress.abort) as abort,
            pytest.raises(RuntimeError, match=f"{workflow} callback failed"),
        ):
            run_workflow()

        abort.assert_called_once_with()
        assert callback.completed == set()

    def test_train_without_training_step_raises_error(self) -> None:
        """Test that training without training step handler raises ValueError."""

        class ModelWithoutTrainStep(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)  # Match input size

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)  # type: ignore[no-any-return]

        model = ModelWithoutTrainStep()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer)
        train_loader = self.create_simple_data_loader()

        with pytest.raises(
            ValueError, match=r"No method decorated with @charge\(Event.TRAIN_STEP\)"
        ):
            battery.train(train_loader)

    @patch("torch_batteries.utils.progress.base.Progress.end_phase")
    def test_train_basic_functionality(self, mock_end_phase: MagicMock) -> None:
        """Test basic train functionality."""
        # Setup mock to return a loss value
        mock_end_phase.return_value = 0.5

        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer)
        train_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        result = battery.train(train_loader, epochs=2, verbose=0)

        assert isinstance(result, dict)
        assert "train_loss" in result
        assert "val_loss" in result
        assert len(result["train_loss"]) == 2  # 2 epochs
        assert len(result["val_loss"]) == 0  # no validation loader

    @patch("torch_batteries.utils.progress.base.Progress.end_phase")
    def test_train_with_validation(self, mock_end_phase: MagicMock) -> None:
        """Test training with validation loader."""
        # Setup mock to return a loss value
        mock_end_phase.return_value = 0.3

        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        battery = Battery(model, optimizer=optimizer)

        train_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)
        val_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        result = battery.train(train_loader, val_loader, epochs=1, verbose=0)

        assert len(result["train_loss"]) == 1
        assert len(result["val_loss"]) == 1

    def test_workflow_lifecycle_logs_use_info_with_arguments(self) -> None:
        """Public workflows log starts and successful completions at INFO."""
        model = SimpleModel()
        loader = self.create_simple_data_loader(batch_size=2, num_samples=4)
        battery = Battery(model, optimizer=optim.SGD(model.parameters(), lr=0.01))

        with (
            patch("torch_batteries.trainer._training.logger.info") as train_info,
            patch("torch_batteries.trainer._evaluation.logger.info") as test_info,
            patch("torch_batteries.trainer._prediction.logger.info") as predict_info,
        ):
            battery.train(loader, epochs=1, verbose=0)
            battery.test(loader, verbose=0)
            battery.predict(loader, verbose=0)

        calls = [
            *(call.args for call in train_info.call_args_list),
            *(call.args for call in test_info.call_args_list),
            *(call.args for call in predict_info.call_args_list),
        ]
        assert (
            "Training started: epochs=%d, train_batches=%d, validation=%s",
            1,
            2,
            False,
        ) in calls
        assert (
            "Training completed: completed_epochs=%d, stopped_early=%s",
            1,
            False,
        ) in calls
        assert ("Testing started: batches=%d", 2) in calls
        assert ("Testing completed",) in calls
        assert ("Prediction started: batches=%d", 2) in calls
        assert ("Prediction completed: outputs=%d", 2) in calls

    def test_train_event_context_includes_metric_history(self) -> None:
        """Test training lifecycle contexts include copied metric history."""
        recorder = ContextRecorder()
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(
            model,
            optimizer=optimizer,
            metrics={"mae": mae},
            callbacks=[recorder],
        )
        train_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)
        val_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        result = battery.train(train_loader, val_loader, epochs=2, verbose=0)

        after_train = recorder.after_train[-1]
        assert after_train["history_train_loss"] == result["train_loss"]
        assert after_train["history_val_loss"] == result["val_loss"]
        assert after_train["history_train_metrics"] == result["train_metrics"]
        assert after_train["history_val_metrics"] == result["val_metrics"]
        assert len(after_train["history_train_loss"]) == 2
        assert len(after_train["history_val_loss"]) == 2
        assert len(after_train["history_train_metrics"]["mae"]) == 2
        assert len(after_train["history_val_metrics"]["mae"]) == 2
        assert isinstance(after_train["train_metrics"]["mae"], float)
        assert isinstance(after_train["val_metrics"]["mae"], float)
        assert [context["epoch"] for context in recorder.before_validation] == [1, 2]
        assert [context["epoch"] for context in recorder.after_validation] == [1, 2]
        assert after_train["epoch"] == 2

    def test_validation_event_context_includes_available_history(self) -> None:
        """Test validation boundary contexts expose history accumulated so far."""
        recorder = ContextRecorder()
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(
            model,
            optimizer=optimizer,
            metrics={"mae": mae},
            callbacks=[recorder],
        )
        train_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)
        val_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        battery.train(train_loader, val_loader, epochs=2, verbose=0)

        assert len(recorder.before_validation[0]["history_train_loss"]) == 1
        assert recorder.before_validation[0]["history_val_loss"] == []
        assert len(recorder.before_validation[1]["history_train_loss"]) == 2
        assert len(recorder.before_validation[1]["history_val_loss"]) == 1
        assert len(recorder.after_validation[-1]["history_train_loss"]) == 2
        assert len(recorder.after_validation[-1]["history_val_loss"]) == 2
        assert isinstance(recorder.after_validation[-1]["val_metrics"]["mae"], float)

    def test_step_event_contexts_include_phase_specific_loss(self) -> None:
        """Test step contexts keep loss alias and add phase-specific loss keys."""
        recorder = ContextRecorder()
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer, callbacks=[recorder])
        train_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)
        val_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        battery.train(train_loader, val_loader, epochs=1, verbose=0)

        train_step_context = recorder.after_train_step[0]
        assert train_step_context["train_loss"] == train_step_context["loss"]
        assert train_step_context["train_metrics"]["loss"] == train_step_context["loss"]

        val_step_context = recorder.after_validation_step[0]
        assert val_step_context["val_loss"] == val_step_context["loss"]
        assert val_step_context["val_metrics"]["loss"] == val_step_context["loss"]

    def test_validate_epoch_without_handler_raises_error(self) -> None:
        """Test validation without validation step handler raises error."""

        class ModelWithoutValidationStep(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)  # Match input size

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)  # type: ignore[no-any-return]

            @charge(Event.TRAIN_STEP)
            def training_step(self, context: EventContext) -> torch.Tensor:
                batch = context["batch"]
                x, y = batch
                pred = self(x)
                return nn.functional.mse_loss(pred, y)

        model = ModelWithoutValidationStep()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer)

        train_loader = self.create_simple_data_loader()
        val_loader = self.create_simple_data_loader()

        with pytest.raises(
            ValueError,
            match=r"No method decorated with @charge\(Event.VALIDATION_STEP\)",
        ):
            battery.train(train_loader, val_loader, epochs=1)

    def test_test_without_handler_raises_error(self) -> None:
        """Test testing without test step handler raises ValueError."""

        class ModelWithoutTestStep(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = ModelWithoutTestStep()
        battery = Battery(model)
        test_loader = self.create_simple_data_loader()

        with pytest.raises(
            ValueError, match=r"No method decorated with @charge\(Event.TEST_STEP\)"
        ):
            battery.test(test_loader)

    @patch("torch_batteries.utils.progress.base.Progress.end_phase")
    def test_test_basic_functionality(self, mock_end_phase: MagicMock) -> None:
        """Test basic test functionality."""
        mock_end_phase.return_value = 0.25

        model = SimpleModel()
        battery = Battery(model)
        test_loader = self.create_simple_data_loader()

        result = battery.test(test_loader, verbose=0)

        assert isinstance(result, dict)
        assert "test_loss" in result
        # Don't assert exact value since it's computed, just assert it's a float
        assert isinstance(result["test_loss"], float)

    def test_test_event_context_includes_phase_specific_loss(self) -> None:
        """Test test contexts keep loss alias and add test_loss."""
        recorder = ContextRecorder()
        model = SimpleModel()
        battery = Battery(model, callbacks=[recorder])
        test_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        battery.test(test_loader, verbose=0)

        test_step_context = recorder.after_test_step[0]
        assert test_step_context["test_loss"] == test_step_context["loss"]
        assert test_step_context["test_metrics"]["loss"] == test_step_context["loss"]

        after_test_context = recorder.after_test[-1]
        assert after_test_context["test_loss"] == after_test_context["loss"]
        assert after_test_context["test_metrics"]["loss"] == after_test_context["loss"]

    def test_automatic_metrics_use_single_forward_per_phase(self) -> None:
        """Automatic metrics reuse predictions returned by each step."""

        class CountingModel(SimpleModel):
            def __init__(self) -> None:
                super().__init__()
                self.forward_calls = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.forward_calls += 1
                return super().forward(x)

        model = CountingModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer, metrics={"mae": mae})
        loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        battery.train(loader, loader, epochs=1, verbose=0)
        assert model.forward_calls == 4

        model.forward_calls = 0
        battery.test(loader, verbose=0)
        assert model.forward_calls == 2

    def test_automatic_metrics_update_batch_norm_once_per_batch(self) -> None:
        """Automatic metrics do not perform extra stateful model forwards."""

        class BatchNormModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.batch_norm = nn.BatchNorm1d(10)
                self.linear = nn.Linear(10, 1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.linear(self.batch_norm(inputs))  # type: ignore[no-any-return]

            @charge(Event.TRAIN_STEP)
            def training_step(self, context: EventContext) -> StepOutput:
                inputs, targets = context["batch"]
                predictions = self(inputs)
                return StepOutput(
                    loss=nn.functional.mse_loss(predictions, targets),
                    predictions=predictions,
                    targets=targets,
                )

        model = BatchNormModel()
        battery = Battery(
            model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            metrics={"mae": mae},
        )
        loader = self.create_simple_data_loader(batch_size=2, num_samples=6)

        battery.train(loader, verbose=0)

        assert model.batch_norm.num_batches_tracked is not None
        assert model.batch_norm.num_batches_tracked.item() == len(loader)

    def test_step_output_supports_dictionary_batches(self) -> None:
        """Automatic metrics do not depend on positional batch fields."""

        class DictionaryDataset(Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                return {
                    "features": torch.full((2,), float(index)),
                    "label": torch.tensor([float(index)]),
                }

        class DictionaryModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 1)

            @charge(Event.TRAIN_STEP)
            def training_step(self, context: EventContext) -> StepOutput:
                batch = context["batch"]
                predictions = self.linear(batch["features"])
                targets = batch["label"]
                return StepOutput(
                    loss=nn.functional.mse_loss(predictions, targets),
                    predictions=predictions,
                    targets=targets,
                )

        model = DictionaryModel()
        battery = Battery(
            model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            metrics={"mae": mae},
        )

        result = battery.train(DataLoader(DictionaryDataset(), batch_size=2), verbose=0)

        assert "mae" in result["train_metrics"]

    def test_step_output_supports_multiple_inputs(self) -> None:
        """Steps can explicitly expose predictions from multi-input batches."""

        class MultiInputModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 1)

            @charge(Event.TRAIN_STEP)
            def training_step(self, context: EventContext) -> StepOutput:
                left, right, targets = context["batch"]
                predictions = self.linear(torch.cat((left, right), dim=1))
                return StepOutput(
                    loss=nn.functional.mse_loss(predictions, targets),
                    predictions=predictions,
                    targets=targets,
                )

        model = MultiInputModel()
        battery = Battery(
            model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            metrics={"mae": mae},
        )
        loader = DataLoader(
            TensorDataset(torch.randn(4, 1), torch.randn(4, 1), torch.randn(4, 1)),
            batch_size=2,
        )

        result = battery.train(loader, verbose=0)

        assert "mae" in result["train_metrics"]

    def test_automatic_metrics_require_explicit_step_output(self) -> None:
        """Legacy step returns cannot drive automatic metrics safely."""

        class LegacyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)

            @charge(Event.TRAIN_STEP)
            def training_step(self, context: EventContext) -> torch.Tensor:
                x, targets = context["batch"]
                return nn.functional.mse_loss(self.linear(x), targets)

        model = LegacyModel()
        battery = Battery(
            model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            metrics={"mae": mae},
        )

        with pytest.raises(ValueError, match="must return StepOutput"):
            battery.train(self.create_simple_data_loader(), verbose=0)

    def test_automatic_metrics_reject_incomplete_step_output(self) -> None:
        """StepOutput must expose predictions and targets for automatic metrics."""

        class IncompleteOutputModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)

            @charge(Event.TRAIN_STEP)
            def training_step(self, context: EventContext) -> StepOutput:
                inputs, targets = context["batch"]
                loss = nn.functional.mse_loss(self.linear(inputs), targets)
                return StepOutput(loss=loss)

        model = IncompleteOutputModel()
        battery = Battery(
            model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            metrics={"mae": mae},
        )

        with pytest.raises(
            ValueError,
            match="StepOutput with predictions and targets",
        ):
            battery.train(self.create_simple_data_loader(), verbose=0)

    def test_manual_step_metrics_override_automatic_metrics(self) -> None:
        """Explicit step metrics retain precedence over automatic metrics."""

        class ManualMetricModel(SimpleModel):
            @charge(Event.TRAIN_STEP)
            def training_step(self, context: EventContext) -> StepOutput:
                x, targets = context["batch"]
                predictions = self(x)
                return StepOutput(
                    loss=nn.functional.mse_loss(predictions, targets),
                    predictions=predictions,
                    targets=targets,
                    metrics={"mae": torch.tensor(123.0)},
                )

        model = ManualMetricModel()
        battery = Battery(
            model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            metrics={"mae": mae},
        )

        result = battery.train(self.create_simple_data_loader(), verbose=0)

        assert result["train_metrics"]["mae"] == [123.0]

    def test_legacy_tuple_metrics_remain_supported(self) -> None:
        """Manual tuple metrics remain valid without automatic metrics."""

        class TupleMetricModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)

            @charge(Event.TRAIN_STEP)
            def training_step(
                self, context: EventContext
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                x, targets = context["batch"]
                loss = nn.functional.mse_loss(self.linear(x), targets)
                return loss, {"manual": torch.tensor(2.0)}

        model = TupleMetricModel()
        battery = Battery(model, optimizer=optim.SGD(model.parameters(), lr=0.01))

        result = battery.train(self.create_simple_data_loader(), verbose=0)

        assert result["train_metrics"]["manual"] == [2.0]

    @pytest.mark.parametrize(
        ("step_result", "error_type", "message"),
        [
            (1.0, TypeError, "loss must be a torch.Tensor"),
            (torch.ones(2), ValueError, "loss must be a scalar tensor"),
            (
                (torch.tensor(1.0), {"bad": torch.ones(2)}),
                ValueError,
                "must be scalar",
            ),
            (
                (torch.tensor(1.0), {"bad": "not-numeric"}),
                TypeError,
                "must be numeric",
            ),
        ],
    )
    def test_invalid_step_results_raise_clear_errors(
        self, step_result: object, error_type: type[Exception], message: str
    ) -> None:
        """Malformed legacy step results raise explicit errors."""

        class InvalidResultModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.parameter = nn.Parameter(torch.tensor(1.0))

            @charge(Event.TRAIN_STEP)
            def training_step(self, _: EventContext) -> object:
                return step_result

        model = InvalidResultModel()
        battery = Battery(model, optimizer=optim.SGD(model.parameters(), lr=0.01))

        with pytest.raises(error_type, match=message):
            battery.train(self.create_simple_data_loader(), verbose=0)

    def test_predict_without_handler_raises_error(self) -> None:
        """Test prediction without predict step handler raises ValueError."""

        class ModelWithoutPredictStep(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = ModelWithoutPredictStep()
        battery = Battery(model)
        data_loader = self.create_simple_data_loader()

        with pytest.raises(
            ValueError, match=r"No method decorated with @charge\(Event.PREDICT_STEP\)"
        ):
            battery.predict(data_loader)

    @patch("torch_batteries.utils.progress.base.Progress.end_phase")
    def test_predict_basic_functionality(self, mock_end_phase: MagicMock) -> None:
        """Test basic predict functionality."""
        mock_end_phase.return_value = float("nan")  # Predict doesn't return loss

        model = SimpleModel()
        battery = Battery(model)
        data_loader = self.create_simple_data_loader(batch_size=2, num_samples=4)

        result = battery.predict(data_loader, verbose=0)

        assert isinstance(result, dict)
        assert "predictions" in result
        assert len(result["predictions"]) == 2  # 4 samples / 2 batch_size = 2 batches

    def test_train_sets_model_to_train_mode(self) -> None:
        """Test that training sets model to train mode."""
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        battery = Battery(model, optimizer=optimizer)

        # Set model to eval mode initially
        model.eval()
        assert not model.training

        train_loader = self.create_simple_data_loader(batch_size=2, num_samples=2)

        with patch(
            "torch_batteries.utils.progress.base.Progress.end_phase"
        ) as mock_end_phase:
            mock_end_phase.return_value = 0.1
            battery.train(train_loader, epochs=1, verbose=0)

        assert model.training

    def test_test_sets_model_to_eval_mode(self) -> None:
        """Test that testing sets model to eval mode."""
        model = SimpleModel()
        battery = Battery(model)

        # Set model to train mode initially
        model.train()
        assert model.training

        test_loader = self.create_simple_data_loader(batch_size=2, num_samples=2)

        with patch(
            "torch_batteries.utils.progress.base.Progress.end_phase"
        ) as mock_end_phase:
            mock_end_phase.return_value = 0.1
            battery.test(test_loader, verbose=0)

        assert not model.training

    def test_predict_sets_model_to_eval_mode(self) -> None:
        """Test that prediction sets model to eval mode."""
        model = SimpleModel()
        battery = Battery(model)

        # Set model to train mode initially
        model.train()
        assert model.training

        data_loader = self.create_simple_data_loader(batch_size=2, num_samples=2)

        with patch(
            "torch_batteries.utils.progress.base.Progress.end_phase"
        ) as mock_end_phase:
            mock_end_phase.return_value = 0.0
            battery.predict(data_loader, verbose=0)

        assert not model.training

    def test_integration_train_and_test(self) -> None:
        """Test integration between training and testing."""
        model = SimpleModel(input_size=5, output_size=1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        battery = Battery(model, optimizer=optimizer)

        # Create simple data
        train_loader = self.create_simple_data_loader(
            batch_size=2, num_samples=4, input_size=5
        )
        test_loader = self.create_simple_data_loader(
            batch_size=2, num_samples=4, input_size=5
        )

        # Train for 1 epoch
        train_result = battery.train(train_loader, epochs=1, verbose=0)

        # Test the trained model
        test_result = battery.test(test_loader, verbose=0)

        assert isinstance(train_result["train_loss"][0], float)
        assert isinstance(test_result["test_loss"], float)

    def test_metrics_property_can_be_reconfigured(self) -> None:
        """Replacing metrics rebuilds the phase metric manager."""
        battery = Battery(SimpleModel())
        original_manager = battery._metric_manager  # noqa: SLF001

        battery.metrics = {"mae": mae}

        assert battery.metrics == {"mae": mae}
        assert battery._metric_manager is not original_manager  # noqa: SLF001

        battery.metrics = None
        assert battery.metrics == {}

    @pytest.mark.parametrize(
        "result",
        [
            (torch.tensor(1.0),),
            (torch.tensor(1.0), []),
            (torch.tensor(1.0), {}, "extra"),
        ],
    )
    def test_malformed_legacy_step_tuple_is_rejected(
        self, result: tuple[object, ...]
    ) -> None:
        """Legacy tuple results must contain exactly loss and metric mapping."""
        battery = Battery(SimpleModel())

        with pytest.raises(TypeError, match="must be \\(loss, metrics_dict\\)"):
            battery._parse_step_result(result, "training")  # noqa: SLF001

    def test_unsized_loader_is_rejected(self) -> None:
        """Workflow loaders must expose their number of batches."""

        class UnsizedLoader:
            def __iter__(self) -> Iterator[object]:
                return iter(())

        battery = Battery(SimpleModel())

        with pytest.raises(ValueError, match="must define its number of batches"):
            battery.test(UnsizedLoader())  # type: ignore[call-overload]

    def test_before_backward_must_preserve_tensor_loss(self) -> None:
        """Callbacks cannot replace the backward loss with a non-tensor."""

        class InvalidBackwardLoss:
            @charge(Event.BEFORE_BACKWARD)
            def replace_loss(self, context: EventContext) -> None:
                context["backward_loss"] = "invalid"  # type: ignore[typeddict-item]

        model = SimpleModel()
        battery = Battery(
            model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            callbacks=[InvalidBackwardLoss()],
        )

        with pytest.raises(
            TypeError,
            match=r"BEFORE_BACKWARD must leave backward_loss as a torch.Tensor",
        ):
            battery.train(self.create_simple_data_loader(), verbose=0)

    def test_stop_request_breaks_before_the_next_epoch(self) -> None:
        """A callback stop request prevents another epoch from starting."""

        class StopAfterFirstEpoch:
            @charge(Event.AFTER_TRAIN_EPOCH)
            def stop(self, context: EventContext) -> None:
                context["battery"].stop_training = True

        model = SimpleModel()
        battery = Battery(
            model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            callbacks=[StopAfterFirstEpoch()],
        )

        result = battery.train(self.create_simple_data_loader(), epochs=3, verbose=0)

        assert len(result["train_loss"]) == 1
        assert battery.stop_training is True

    def test_validation_epoch_defensively_checks_for_handler(self) -> None:
        """The epoch executor validates its handler independently."""

        class ModelWithoutValidation(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)

        battery = Battery(ModelWithoutValidation())

        with pytest.raises(
            ValueError,
            match=r"No method decorated with @charge\(Event.VALIDATION_STEP\)",
        ):
            battery._validate_epoch(  # noqa: SLF001
                self.create_simple_data_loader(),
                SilentProgress(),
                epoch=1,
            )
