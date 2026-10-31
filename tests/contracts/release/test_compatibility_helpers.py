"""Validate portable CI helpers without installing into the developer environment."""

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_dependency_install_reads_project_requirements_without_upgrading_pip(
    tmp_path: Path,
) -> None:
    helper = importlib.import_module("scripts.ci.compatibility")
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools>=77.0.3", "wheel"]\n'
        '[dependency-groups]\ntest=["pytest>=7", "pytest-cov>=4", "pytest-mock>=3"]\n',
        encoding="utf-8",
    )
    with patch.object(helper.subprocess, "run") as run:
        helper.install_test_dependencies(tmp_path)
    command = run.call_args.args[0]
    assert command[:4] == [sys.executable, "-m", "pip", "install"]
    assert {
        "pytest>=7",
        "pytest-cov>=4",
        "pytest-mock>=3",
        "setuptools>=77.0.3",
        "wheel",
    } <= set(command)
    assert "--group" not in command
    assert "--upgrade" not in command
    assert "pip" not in command[4:]


def test_wheel_install_uses_the_selected_interpreter_and_preserves_dependencies(
    tmp_path: Path,
) -> None:
    helper = importlib.import_module("scripts.ci.compatibility")
    wheel = tmp_path / "package.whl"
    wheel.touch()
    with patch.object(helper.subprocess, "run") as run:
        helper.install_wheel(tmp_path)
    command = run.call_args.args[0]
    assert command[:4] == [sys.executable, "-m", "pip", "install"]
    assert "--no-deps" in command
    assert str(wheel) in command
    assert "-e" not in command


@pytest.mark.parametrize("count", [0, 2])
def test_ambiguous_wheel_build_fails_without_installing(
    tmp_path: Path, count: int
) -> None:
    helper = importlib.import_module("scripts.ci.compatibility")
    for index in range(count):
        (tmp_path / f"package{index}.whl").touch()
    with (
        patch.object(helper.subprocess, "run") as run,
        pytest.raises(ValueError, match="exactly one wheel"),
    ):
        helper.install_wheel(tmp_path)
    run.assert_not_called()


def test_environment_report_records_native_interpreter_and_package_versions(
    tmp_path: Path,
) -> None:
    helper = importlib.import_module("scripts.ci.compatibility")
    helper.record_environment(tmp_path / "environment.json")
    payload = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    assert payload["executable"] == sys.executable
    assert payload["machine"]
    assert "torch" in payload["packages"]
