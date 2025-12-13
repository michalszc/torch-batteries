"""Tests for torch_batteries.events module."""

import torch
from torch import nn

from torch_batteries.events import Event, EventHandler, charge


class TestChargeDecorator:
    """Test cases for charge decorator."""

    def test_charge_decorator_sets_attribute(self) -> None:
        """Test that charge decorator sets _torch_batteries_event attribute."""

        @charge(Event.TRAIN_STEP)
        def training_step(self: nn.Module, batch: torch.Tensor) -> torch.Tensor:
            return torch.tensor(1.0)

        assert hasattr(training_step, "_torch_batteries_event")
        assert training_step._torch_batteries_event == Event.TRAIN_STEP  # noqa: SLF001 # type: ignore[attr-defined]

    def test_charge_decorator_preserves_function(self) -> None:
        """Test that charge decorator preserves original function."""

        def original_func(self: nn.Module, batch: str) -> str:
            return "original_result"

        decorated = charge(Event.VALIDATION_STEP)(original_func)

        # Function should still work
        assert decorated(nn.Module(), "test_batch") == "original_result"
        assert decorated._torch_batteries_event == Event.VALIDATION_STEP  # type: ignore[attr-defined] # noqa: SLF001

    def test_charge_decorator_with_different_events(self) -> None:
        """Test charge decorator works with different events."""

        @charge(Event.TEST_STEP)
        def test_func() -> None:
            pass

        @charge(Event.PREDICT_STEP)
        def predict_func() -> None:
            pass

        assert test_func._torch_batteries_event == Event.TEST_STEP  # type: ignore[attr-defined] # noqa: SLF001
        assert predict_func._torch_batteries_event == Event.PREDICT_STEP  # type: ignore[attr-defined] # noqa: SLF001


class DummyModel(nn.Module):
    """Dummy model for testing EventHandler."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    @charge(Event.TRAIN_STEP)
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.3)

    def regular_method(self, batch: str) -> str:
        return "not decorated"


class TestEventHandler:
    """Test cases for EventHandler class."""

    def test_init_discovers_handlers(self) -> None:
        """Test that EventHandler discovers decorated methods."""
        model = DummyModel()
        handler = EventHandler(model)

        assert handler.has_handler(Event.TRAIN_STEP)
        assert handler.has_handler(Event.VALIDATION_STEP)
        assert not handler.has_handler(Event.TEST_STEP)
        assert not handler.has_handler(Event.PREDICT_STEP)

    def test_call_existing_handler(self) -> None:
        """Test calling an existing event handler."""
        model = DummyModel()
        handler = EventHandler(model)

        result = handler.call(Event.TRAIN_STEP, "test_batch")

        assert torch.equal(result, torch.tensor(0.0))

    def test_call_nonexistent_handler_returns_none(self) -> None:
        """Test calling a non-existent event handler returns None."""
        model = DummyModel()
        handler = EventHandler(model)

        result = handler.call(Event.TEST_STEP, "test_batch")
        assert result is None

    def test_has_handler_returns_correct_boolean(self) -> None:
        """Test has_handler returns correct boolean values."""
        model = DummyModel()
        handler = EventHandler(model)

        assert handler.has_handler(Event.TRAIN_STEP) is True
        assert handler.has_handler(Event.VALIDATION_STEP) is True
        assert handler.has_handler(Event.TEST_STEP) is False
        assert handler.has_handler(Event.PREDICT_STEP) is False

    def test_handler_calls_method_with_correct_arguments(self) -> None:
        """Test that handler calls method with correct arguments."""

        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.call_args: dict[str, torch.Tensor] = {}

            @charge(Event.TRAIN_STEP)
            def training_step(self, batch: dict) -> torch.Tensor:
                self.call_args = batch
                return torch.tensor(1.0)

        model = TestModel()
        handler = EventHandler(model)

        test_batch = {"input": torch.randn(2, 3), "target": torch.randn(2, 1)}
        handler.call(Event.TRAIN_STEP, test_batch)

        assert model.call_args == test_batch

    def test_empty_model_has_no_handlers(self) -> None:
        """Test that model with no decorated methods has no handlers."""

        class EmptyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def regular_method(self) -> str:
                return "not decorated"

        model = EmptyModel()
        handler = EventHandler(model)

        assert not handler.has_handler(Event.TRAIN_STEP)
        assert not handler.has_handler(Event.VALIDATION_STEP)
        assert not handler.has_handler(Event.TEST_STEP)
        assert not handler.has_handler(Event.PREDICT_STEP)

    def test_multiple_methods_same_event(self) -> None:
        """Test that only last method is registered for same event."""

        class MultipleMethodsModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            @charge(Event.TRAIN_STEP)
            def first_training_step(self, batch: str) -> torch.Tensor:
                return torch.tensor(1.0)

            @charge(Event.TRAIN_STEP)
            def second_training_step(self, batch: str) -> torch.Tensor:
                return torch.tensor(2.0)

        model = MultipleMethodsModel()
        handler = EventHandler(model)

        result = handler.call(Event.TRAIN_STEP, "batch")
        # Should call the last registered method
        assert torch.equal(result, torch.tensor(2.0))

    def test_handler_works_with_inheritance(self) -> None:
        """Test that event handler works with inherited methods."""

        class BaseModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            @charge(Event.TRAIN_STEP)
            def training_step(self, batch: str) -> torch.Tensor:
                return torch.tensor(1.0)

        class ChildModel(BaseModel):
            def __init__(self) -> None:
                super().__init__()

            @charge(Event.VALIDATION_STEP)
            def validation_step(self, batch: str) -> torch.Tensor:
                return torch.tensor(2.0)

        model = ChildModel()
        handler = EventHandler(model)

        assert handler.has_handler(Event.TRAIN_STEP)
        assert handler.has_handler(Event.VALIDATION_STEP)

        train_result = handler.call(Event.TRAIN_STEP, "batch")
        val_result = handler.call(Event.VALIDATION_STEP, "batch")

        assert torch.equal(train_result, torch.tensor(1.0))
        assert torch.equal(val_result, torch.tensor(2.0))
