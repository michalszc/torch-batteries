"""Integration tests for the release-version validation script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[3]
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

    fake_pypi(tmp_path, '{"info":{"version":"0.8.0"}}')
    return tmp_path


def fake_pypi(project: Path, body: str, status: int = 200) -> None:
    """Replace curl with one that writes `body` to its -o file and reports `status`."""
    curl = project / "bin" / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\n"
        'while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done\n'
        f"printf '%s' '{body}' > \"$out\"\n"
        f"printf '%s' '{status}'\n",
        encoding="utf-8",
    )
    curl.chmod(0o755)


def validate(
    project: Path, *arguments: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run validation in a fixture project without accessing PyPI."""
    environment = os.environ.copy()
    environment.pop("RELEASE_PROJECT", None)
    if cwd is not None:
        environment["RELEASE_PROJECT"] = str(project)
    environment["PATH"] = ":".join(
        (str(project / "bin"), str(PROJECT_ROOT / ".venv" / "bin"), environment["PATH"])
    )
    return subprocess.run(
        ["bash", str(VALIDATOR), *arguments],
        cwd=cwd or project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_accepts_version_in_release_notes(version_project: Path) -> None:
    result = validate(version_project)

    assert result.returncode == 0
    assert "included exactly once in release notes" in result.stdout


def test_validates_release_project_from_another_directory(
    version_project: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    result = validate(version_project, cwd=tmp_path_factory.mktemp("tooling"))

    assert result.returncode == 0
    assert f"All checks passed for version {VERSION}" in result.stdout


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


def test_default_validation_does_not_query_pypi(version_project: Path) -> None:
    (version_project / "bin" / "curl").write_text("#!/usr/bin/env bash\nexit 7\n")

    result = validate(version_project)

    assert result.returncode == 0
    assert "PyPI" not in result.stdout


def test_unpublished_validation_accepts_new_version(version_project: Path) -> None:
    result = validate(version_project, "--unpublished")

    assert result.returncode == 0
    assert f"Version {VERSION} is newer than PyPI (0.8.0)" in result.stdout


@pytest.mark.parametrize("pypi_version", [VERSION, "0.10.0", "1.0.0"])
def test_unpublished_validation_rejects_version_not_newer_than_pypi(
    version_project: Path, pypi_version: str
) -> None:
    fake_pypi(version_project, f'{{"info":{{"version":"{pypi_version}"}}}}')

    result = validate(version_project, "--unpublished")

    assert result.returncode != 0
    assert f"not newer than PyPI ({pypi_version})" in result.stdout


def test_unpublished_validation_accepts_first_release(version_project: Path) -> None:
    fake_pypi(version_project, '{"message":"Not Found"}', status=404)

    result = validate(version_project, "--unpublished")

    assert result.returncode == 0
    assert "not found on PyPI (first release?)" in result.stdout


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (503, "Service Unavailable", "PyPI returned HTTP 503"),
        (429, "Too Many Requests", "PyPI returned HTTP 429"),
        (200, "<html>maintenance</html>", "Could not read the latest version"),
    ],
)
def test_unpublished_validation_fails_closed_on_pypi_errors(
    version_project: Path, status: int, body: str, error: str
) -> None:
    fake_pypi(version_project, body, status=status)

    result = validate(version_project, "--unpublished")

    assert result.returncode != 0
    assert error in result.stdout
    assert "first release" not in result.stdout


def test_unpublished_validation_fails_when_pypi_is_unreachable(
    version_project: Path,
) -> None:
    (version_project / "bin" / "curl").write_text("#!/usr/bin/env bash\nexit 7\n")

    result = validate(version_project, "--unpublished")

    assert result.returncode != 0
    assert "Could not reach PyPI" in result.stdout
