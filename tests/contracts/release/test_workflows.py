"""Contracts for deployment and release workflow boundaries."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).parents[3]
SHA = "a" * 40


def workflow(name: str) -> dict[str, Any]:
    """Read YAML without interpreting GitHub's `on` key as a boolean."""
    result: dict[str, Any] = yaml.load(
        (ROOT / ".github" / "workflows" / f"{name}.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    return result


def permits(guard: str, **fields: str) -> bool:
    """Evaluate the conjunction of equality guards used by these workflows."""
    clauses = guard.removeprefix("${{").removesuffix("}}").strip().split("&&")
    results = []
    for clause in clauses:
        match = re.fullmatch(
            r"\s*github\.event\.workflow_run\.([\w.]+) == "
            r"(?:'([^']+)'|(github\.repository))\s*",
            clause,
        )
        assert match is not None, f"Unsupported workflow guard: {clause}"
        field, literal, repository = match.groups()
        expected = "owner/repo" if repository else literal
        results.append(fields[field] == expected)
    return all(results)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("conclusion", "success", True),
        ("conclusion", "failure", False),
        ("conclusion", "cancelled", False),
        ("conclusion", "skipped", False),
        ("event", "pull_request", False),
        ("head_branch", "feature/example", False),
        ("head_repository.full_name", "fork/repo", False),
    ],
)
def test_cd_guards(field: str, value: str, *, expected: bool) -> None:
    fields = {
        "conclusion": "success",
        "event": "push",
        "head_branch": "master",
        "head_repository.full_name": "owner/repo",
    }
    fields[field] = value
    assert permits(workflow("cd")["jobs"]["publish"]["if"], **fields) is expected


def test_cd_records_identity_before_publishing(tmp_path: Path) -> None:
    cd = workflow("cd")
    assert cd["on"]["workflow_run"] == {
        "workflows": ["CI"],
        "branches": ["master"],
        "types": ["completed"],
    }
    steps = cd["jobs"]["publish"]["steps"]
    assert steps[0]["with"]["ref"] == "${{ github.event.workflow_run.head_sha }}"
    names = [step.get("name", "") for step in steps]
    record = names.index("Record deployment identity")
    upload = names.index("Upload deployment identity")
    publish = names.index("Publish to PyPI")
    assert record < upload < publish
    assert steps[upload]["with"] == {
        "name": "release-metadata",
        "path": "release-metadata.json",
        "if-no-files-found": "error",
    }
    assert steps[names.index("Build package")]["run"] == "make build"
    assert steps[names.index("Check build")]["run"] == "make check-build"
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "torch-batteries"\nversion = "1.0.0"\n',
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    git = binary / "git"
    git.write_text(f"#!/bin/sh\nprintf '%s\\n' '{SHA}'\n")
    git.chmod(0o755)
    assert steps[record]["run"] == "make release-metadata"
    assert (
        steps[record]["env"]["DEPLOYED_SHA"]
        == "${{ github.event.workflow_run.head_sha }}"
    )
    (tmp_path / "Makefile").write_text((ROOT / "Makefile").read_text())
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "release.py").write_text((ROOT / "scripts" / "release.py").read_text())
    environment = {**os.environ, "DEPLOYED_SHA": SHA}
    environment["PATH"] = f"{binary}:{ROOT / '.venv' / 'bin'}:{environment['PATH']}"
    result = subprocess.run(
        ["make", "release-metadata"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "release-metadata.json").read_text()) == {
        "sha": SHA,
        "name": "torch-batteries",
        "version": "1.0.0",
    }
    environment["DEPLOYED_SHA"] = "b" * 40
    result = subprocess.run(
        ["make", "release-metadata"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Checkout does not match" in result.stderr

    # A local invocation needs no CI environment or network access.
    environment.pop("DEPLOYED_SHA")
    result = subprocess.run(
        ["make", "release-metadata", "RELEASE_METADATA=local.json"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "local.json").read_text())["sha"] == SHA


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("conclusion", "success", True),
        ("conclusion", "failure", False),
        ("conclusion", "cancelled", False),
        ("conclusion", "skipped", False),
        ("head_branch", "feature/example", False),
        ("head_repository.full_name", "fork/repo", False),
    ],
)
def test_release_guards(field: str, value: str, *, expected: bool) -> None:
    fields = {
        "conclusion": "success",
        "head_branch": "master",
        "head_repository.full_name": "owner/repo",
    }
    fields[field] = value
    assert permits(workflow("release")["jobs"]["release"]["if"], **fields) is expected


def test_release_uses_metadata_from_exact_cd_run() -> None:
    release = workflow("release")
    assert release["on"] == {
        "workflow_run": {
            "workflows": ["CD"],
            "branches": ["master"],
            "types": ["completed"],
        },
    }
    assert release["concurrency"]["cancel-in-progress"] == "false"
    job = release["jobs"]["release"]
    assert job["permissions"] == {"actions": "read", "contents": "write"}
    steps = job["steps"]
    names = [step["name"] for step in steps]
    download = names.index("Download deployment identity")
    select = names.index("Select deployed commit")
    checkout = names.index("Check out deployed commit")
    publish = names.index("Publish GitHub release")
    assert download < select < checkout < publish
    assert steps[download]["with"]["run-id"] == "${{ github.event.workflow_run.id }}"
    assert steps[download]["with"]["repository"] == "${{ github.repository }}"
    assert steps[download]["with"]["name"] == "release-metadata"
    assert steps[checkout]["with"]["ref"] == "${{ steps.deployment.outputs.sha }}"
    assert steps[checkout]["with"]["path"] == "deployed"
    assert steps[publish]["env"]["RELEASE_PROJECT"] == "deployed"
    assert steps[publish]["env"]["GH_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert steps[select]["run"] == "make release-select-commit"
    assert steps[publish]["run"] == "make release"
    assert not any("pip install" in step.get("run", "") for step in steps)


def test_ci_checks_release_automation_changes() -> None:
    ci = workflow("ci")
    for event in ("push", "pull_request"):
        assert {
            ".github/workflows/cd.yml",
            ".github/workflows/release.yml",
            "scripts/release.py",
            "documentation/release-notes.md",
        } <= set(ci["on"][event]["paths"])
    checks = ci["jobs"]["quality-checks"]["strategy"]["matrix"]["check"]
    assert {"lint", "format-check", "type-check", "test"} <= set(checks)
    makefile = (ROOT / "Makefile").read_text()
    assert "ruff check src/ tests/ scripts/" in makefile
    assert "ruff format --diff src/ tests/ scripts/" in makefile
    assert "mypy src/torch_batteries/ tests/ scripts/" in makefile
    assert "pytest tests/" in makefile
