"""Tests for the resumable callback contract."""

import pytest

from torch_batteries.callbacks import Callback


def test_callback_default_state_round_trip() -> None:
    callback = Callback()

    assert callback.state_dict() == {}
    callback.load_state_dict({})


def test_callback_warns_when_ignoring_state(caplog: pytest.LogCaptureFixture) -> None:
    callback = Callback()

    callback.load_state_dict({"counter": 1})

    assert "ignored unexpected state keys" in caplog.text
