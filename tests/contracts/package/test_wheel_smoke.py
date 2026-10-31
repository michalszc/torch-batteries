"""Run the compatibility smoke workflow using the existing local environment."""

import importlib
from pathlib import Path

import pytest


def test_smoke_workflows_cover_spawn_and_resumption(tmp_path: Path) -> None:
    smoke = importlib.import_module("scripts.ci.smoke_installed_package")
    smoke.exercise_workflows(tmp_path)


def test_smoke_rejects_an_import_from_the_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = importlib.import_module("scripts.ci.smoke_installed_package")
    source = Path(__file__).parents[3] / "src" / "torch_batteries" / "__init__.py"
    monkeypatch.setattr(smoke.torch_batteries, "__file__", str(source))
    with pytest.raises(AssertionError, match="Imported source checkout"):
        smoke.main()
