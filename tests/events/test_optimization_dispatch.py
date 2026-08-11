"""Tests for optimization-event dispatch modes."""

from collections.abc import Generator
from contextlib import contextmanager

import pytest
from torch import nn

from torch_batteries import Event, EventContext, OptimizationStep, charge
from torch_batteries.events import EventHandler


@contextmanager
def _recording_context(name: str, calls: list[str]) -> Generator[None]:
    calls.append(f"enter:{name}")
    try:
        yield
    finally:
        calls.append(f"exit:{name}")


class _BroadcastModel(nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    @charge(Event.BEFORE_BACKWARD)
    def before_backward(self, context: EventContext) -> None:
        del context
        self.calls.append("model")


class _BroadcastCallback:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    @charge(Event.BEFORE_BACKWARD)
    def before_backward(self, context: EventContext) -> None:
        del context
        self.calls.append(self.name)


def test_broadcast_order_is_model_then_callback_list() -> None:
    calls: list[str] = []
    handler = EventHandler(
        _BroadcastModel(calls),
        callbacks=[
            _BroadcastCallback("callback-1", calls),
            _BroadcastCallback("callback-2", calls),
        ],
    )

    handler.call(Event.BEFORE_BACKWARD, EventContext())

    assert calls == ["model", "callback-1", "callback-2"]


class _ContextModel(nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    @charge(Event.STEP_EXECUTION_CONTEXT)
    def execution_context(self, context: EventContext) -> object:
        del context
        return _recording_context("model", self.calls)


class _ContextCallback:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    @charge(Event.STEP_EXECUTION_CONTEXT)
    def execution_context(self, context: EventContext) -> object:
        del context
        return _recording_context(self.name, self.calls)


def _fail_in_execution_context(handler: EventHandler, calls: list[str]) -> None:
    with handler.execution_context(
        Event.STEP_EXECUTION_CONTEXT,
        EventContext(),
    ):
        calls.append("body")
        raise RuntimeError


def test_execution_contexts_exit_in_reverse_order_on_failure() -> None:
    calls: list[str] = []
    handler = EventHandler(
        _ContextModel(calls),
        callbacks=[
            _ContextCallback("callback-1", calls),
            _ContextCallback("callback-2", calls),
        ],
    )

    with pytest.raises(RuntimeError):
        _fail_in_execution_context(handler, calls)

    assert calls == [
        "enter:model",
        "enter:callback-1",
        "enter:callback-2",
        "body",
        "exit:callback-2",
        "exit:callback-1",
        "exit:model",
    ]


class _ExclusiveOwner:
    @charge(Event.CONFIGURE_TRAIN_STEP)
    def configure(self, context: EventContext) -> OptimizationStep:
        del context
        return OptimizationStep()


class _ExclusiveModel(nn.Module):
    @charge(Event.CONFIGURE_TRAIN_STEP)
    def configure(self, context: EventContext) -> OptimizationStep:
        del context
        return OptimizationStep()


def test_model_callback_exclusive_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match="accepts exactly one handler"):
        EventHandler(_ExclusiveModel(), callbacks=[_ExclusiveOwner()])


def test_callback_callback_exclusive_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match="accepts exactly one handler"):
        EventHandler(
            nn.Linear(1, 1),
            callbacks=[_ExclusiveOwner(), _ExclusiveOwner()],
        )


class _MultipleExclusiveModel(nn.Module):
    @charge(Event.CONFIGURE_TRAIN_STEP)
    def configure_first(self, context: EventContext) -> OptimizationStep:
        del context
        return OptimizationStep()

    @charge(Event.CONFIGURE_TRAIN_STEP)
    def configure_second(self, context: EventContext) -> OptimizationStep:
        del context
        return OptimizationStep()


def test_model_model_exclusive_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"model\.configure_first"):
        EventHandler(_MultipleExclusiveModel())


class _InvalidContext:
    @charge(Event.STEP_EXECUTION_CONTEXT)
    def execution_context(self, context: EventContext) -> object:
        del context
        return "not-a-context-manager"


def test_invalid_execution_context_return_is_rejected() -> None:
    handler = EventHandler(nn.Linear(1, 1), callbacks=[_InvalidContext()])

    with (
        pytest.raises(TypeError, match="must return context managers"),
        (
            handler.execution_context(
                Event.STEP_EXECUTION_CONTEXT,
                EventContext(),
            )
        ),
    ):
        pass


class _InvalidExecutor:
    @charge(Event.BACKWARD)
    def backward(self, context: EventContext) -> bool:
        del context
        return True


def test_executor_must_return_none() -> None:
    handler = EventHandler(nn.Linear(1, 1), callbacks=[_InvalidExecutor()])

    with pytest.raises(TypeError, match="must return None"):
        handler.execute(Event.BACKWARD, EventContext())


def test_provider_uses_handler_or_default() -> None:
    default = OptimizationStep()
    empty = EventHandler(nn.Linear(1, 1))
    configured = EventHandler(nn.Linear(1, 1), callbacks=[_ExclusiveOwner()])

    assert (
        empty.provide(
            Event.CONFIGURE_TRAIN_STEP,
            EventContext(),
            default=default,
        )
        is default
    )
    assert (
        configured.provide(
            Event.CONFIGURE_TRAIN_STEP,
            EventContext(),
            default=default,
        )
        == OptimizationStep()
    )


def test_provider_rejects_malformed_internal_registration() -> None:
    """Provider dispatch defensively requires one list-based registration."""
    handler = EventHandler(nn.Linear(1, 1))
    handler._event_handlers[Event.CONFIGURE_TRAIN_STEP] = lambda: None  # noqa: SLF001

    with pytest.raises(ValueError, match="requires one provider handler"):
        handler.provide(
            Event.CONFIGURE_TRAIN_STEP,
            EventContext(),
            default=OptimizationStep(),
        )


def test_append_rejects_model_specific_internal_registration() -> None:
    """Broadcast discovery cannot append to a model-specific callable slot."""
    handler = EventHandler(nn.Linear(1, 1))
    handler._event_handlers[Event.BEFORE_TRAIN] = lambda: None  # noqa: SLF001

    with pytest.raises(TypeError, match="Cannot register multiple handlers"):
        handler._append_handler(  # noqa: SLF001
            Event.BEFORE_TRAIN,
            lambda _: None,
            "callback.invalid",
        )


def test_executor_rejects_multiple_internal_registrations() -> None:
    """Executor dispatch defensively rejects ambiguous registrations."""
    handler = EventHandler(nn.Linear(1, 1))
    handler._event_handlers[Event.BACKWARD] = [  # noqa: SLF001
        lambda: None,
        lambda: None,
    ]

    with pytest.raises(ValueError, match="requires one executor handler"):
        handler.execute(Event.BACKWARD, EventContext())
