"""Failure-path tests for atomic checkpoint metadata writes."""

from pathlib import Path

import pytest

from torch_batteries.callbacks._checkpoint_files import _write_json


def test_failed_json_serialization_preserves_existing_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "ranking.json"
    destination.write_text("previous ranking", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON compliant"):
        _write_json({"invalid": float("inf")}, destination)
    assert destination.read_text(encoding="utf-8") == "previous ranking"
    assert list(tmp_path.iterdir()) == [destination]
