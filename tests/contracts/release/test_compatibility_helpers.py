"""Validate portable CI helpers without installing into the developer environment."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

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


def test_wheel_install_resolves_dependencies_and_preserves_installed_cpu_torch(
    tmp_path: Path,
) -> None:
    helper = importlib.import_module("scripts.ci.compatibility")
    wheel = tmp_path / "package.whl"
    wheel.touch()
    with patch.object(helper.subprocess, "run") as run:
        helper.install_wheel(tmp_path)
    command = run.call_args.args[0]
    assert command[:4] == [sys.executable, "-m", "pip", "install"]
    assert "--no-deps" not in command
    assert "--force-reinstall" not in command
    assert str(wheel) in command
    assert "-e" not in command


def test_wheel_install_does_not_silently_ignore_a_missing_dependency(
    tmp_path: Path,
) -> None:
    """Exercise pip's resolver offline in dry-run mode without installing anything."""
    helper = importlib.import_module("scripts.ci.compatibility")
    wheel = tmp_path / "ci_dependency_fixture-1.0.0-py3-none-any.whl"
    metadata = "ci_dependency_fixture-1.0.0.dist-info"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{metadata}/METADATA",
            "Metadata-Version: 2.1\nName: ci-dependency-fixture\nVersion: 1.0.0\n"
            "Requires-Dist: torch-batteries-ci-missing-dependency==1.0.0\n",
        )
        archive.writestr(f"{metadata}/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        archive.writestr(f"{metadata}/RECORD", "")
    run = subprocess.run

    def dry_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        return run(
            [
                *command[:4],
                "--dry-run",
                "--no-index",
                "--ignore-installed",
                *command[4:],
            ],
            check=check,
            capture_output=True,
            text=True,
        )

    with (
        patch.object(helper.subprocess, "run", side_effect=dry_run),
        pytest.raises(subprocess.CalledProcessError) as error,
    ):
        helper.install_wheel(tmp_path)
    assert "torch-batteries-ci-missing-dependency" in error.value.stderr


def test_cpu_wheel_download_and_dependency_install_use_separate_indexes() -> None:
    helper = importlib.import_module("scripts.ci.compatibility")
    commands: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> None:
        commands.append(command)
        if "download" in command:
            destination = Path(command[command.index("--dest") + 1])
            (destination / "torch-2.15.0+cpu-cp313-cp313-win_arm64.whl").touch()

    with patch.object(helper.subprocess, "run", side_effect=run):
        helper.install_cpu_torch()
    download, install = commands
    assert download[:4] == [sys.executable, "-m", "pip", "download"]
    assert "--no-deps" in download
    assert "--only-binary=:all:" in download
    assert "https://download.pytorch.org/whl/cpu" in download
    assert install[:4] == [sys.executable, "-m", "pip", "install"]
    assert "win_arm64.whl" in install[-1]
    assert "--index-url" not in install
    assert "--only-binary=:all:" not in install
    assert "--no-deps" not in install
    assert not Path(install[-1]).parent.exists()


def test_failed_cpu_download_does_not_attempt_installation() -> None:
    helper = importlib.import_module("scripts.ci.compatibility")
    with (
        patch.object(
            helper.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["pip"]),
        ) as run,
        pytest.raises(subprocess.CalledProcessError),
    ):
        helper.install_cpu_torch()
    assert run.call_count == 1


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
