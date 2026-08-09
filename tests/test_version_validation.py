"""Integration tests for the release-version validation script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_version.sh"
VERSION = "0.9.0"


@pytest.fixture
def version_project(tmp_path: Path) -> Path:
    """Create a minimal project with deterministic local PyPI output."""
    package = tmp_path / "src" / "torch_batteries"
    documentation = tmp_path / "documentation"
    fake_bin = tmp_path / "bin"
    package.mkdir(parents=True)
    documentation.mkdir()
    fake_bin.mkdir()

    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "torch-batteries"\nversion = "{VERSION}"\n',
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        f'__version__ = "{VERSION}"\n',
        encoding="utf-8",
    )
    (documentation / "release-notes.md").write_text(
        f"# Release Notes\n\n## {VERSION} — Current\n",
        encoding="utf-8",
    )

    curl = fake_bin / "curl"
    curl.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' \'{"info":{"version":"0.8.0"}}\'\n',
        encoding="utf-8",
    )
    curl.chmod(0o755)
    return tmp_path


def validate(project: Path) -> subprocess.CompletedProcess[str]:
    """Run validation in a fixture project without accessing PyPI."""
    environment = os.environ.copy()
    environment["PATH"] = ":".join(
        (str(project / "bin"), str(PROJECT_ROOT / ".venv" / "bin"), environment["PATH"])
    )
    return subprocess.run(
        ["bash", str(VALIDATOR)],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_accepts_version_in_release_notes(version_project: Path) -> None:
    result = validate(version_project)

    assert result.returncode == 0
    assert "included exactly once in release notes" in result.stdout


def test_rejects_missing_release_version(version_project: Path) -> None:
    release_notes = version_project / "documentation" / "release-notes.md"
    release_notes.write_text(
        "# Release Notes\n\n## 0.8.0 — Previous\n",
        encoding="utf-8",
    )

    result = validate(version_project)

    assert result.returncode != 0
    assert f"Version {VERSION} is missing" in result.stdout


def test_rejects_duplicate_release_version(version_project: Path) -> None:
    release_notes = version_project / "documentation" / "release-notes.md"
    release_notes.write_text(
        f"# Release Notes\n\n## {VERSION} — First\n\n## {VERSION} — Duplicate\n",
        encoding="utf-8",
    )

    result = validate(version_project)

    assert result.returncode != 0
    assert f"Version {VERSION} appears multiple times" in result.stdout
