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
    assert permits(workflow("cd")["jobs"]["build"]["if"], **fields) is expected


def test_cd_uploads_build_without_publishing(tmp_path: Path) -> None:
    cd = workflow("cd")
    assert cd["on"]["workflow_run"] == {
        "workflows": ["CI"],
        "branches": ["master"],
        "types": ["completed"],
    }
    job = cd["jobs"]["build"]
    steps = job["steps"]
    head_sha = "${{ github.event.workflow_run.head_sha }}"
    assert steps[0]["with"]["ref"] == head_sha
    names = [step.get("name", "") for step in steps]
    check = names.index("Check build")
    record = names.index("Record deployment identity")
    upload_build = names.index("Upload package build")
    upload_identity = names.index("Upload deployment identity")
    # The identity is uploaded last, so finding it implies a complete build.
    assert check < record < upload_build < upload_identity == len(steps) - 1
    assert steps[upload_build]["with"] == {
        "name": f"dist-{head_sha}",
        "path": "dist/",
        "if-no-files-found": "error",
        "overwrite": "true",
    }
    assert steps[upload_identity]["with"] == {
        "name": f"release-metadata-{head_sha}",
        "path": "release-metadata.json",
        "if-no-files-found": "error",
        "overwrite": "true",
    }
    assert "Publish to PyPI" not in names
    assert "secrets." not in yaml.dump(job)
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


def test_release_is_manual_and_limited_to_master() -> None:
    release = workflow("release")
    assert release["on"] == {
        "workflow_dispatch": {
            "inputs": {
                "commit": {
                    "description": (
                        "Master commit to release (SHA); default: master HEAD"
                    ),
                    "required": "false",
                    "default": "",
                    "type": "string",
                },
            },
        },
    }
    assert release["concurrency"]["cancel-in-progress"] == "false"
    publish = release["jobs"]["publish"]
    assert publish["if"] == "${{ github.ref == 'refs/heads/master' }}"
    assert publish["environment"]["name"] == "pypi"
    assert publish["permissions"] == {"actions": "read", "contents": "read"}
    assert release["jobs"]["github-release"]["needs"] == "publish"
    assert release["jobs"]["github-release"]["permissions"] == {
        "actions": "read",
        "contents": "write",
    }


def test_release_resolves_commit_on_master_first() -> None:
    jobs = workflow("release")["jobs"]
    steps = jobs["publish"]["steps"]
    names = [step["name"] for step in steps]
    master = names.index("Check out master")
    resolve = names.index("Resolve release commit")
    checkout = names.index("Check out release commit")
    assert master < resolve < checkout < names.index("Find CD build")
    assert steps[master]["with"]["ref"] == "${{ github.sha }}"
    assert steps[master]["with"]["fetch-depth"] == "0"
    assert steps[resolve]["run"] == "make release-resolve-commit"
    assert steps[resolve]["env"]["RELEASE_COMMIT"] == "${{ inputs.commit }}"
    sha = "${{ steps.commit.outputs.sha }}"
    assert steps[checkout]["with"]["ref"] == sha
    assert jobs["publish"]["outputs"]["sha"] == sha
    # Release tooling stays at master's root; the release commit is only data.
    for job, released in (
        ("publish", sha),
        ("github-release", "${{ needs.publish.outputs.sha }}"),
    ):
        project = jobs[job]["env"]["RELEASE_PROJECT"]
        checkouts = [
            step["with"]
            for step in jobs[job]["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        ]
        assert checkouts[0]["ref"] == "${{ github.sha }}"
        assert "path" not in checkouts[0]
        assert {"ref": released, "path": project}.items() <= checkouts[1].items()
        assert len(checkouts) == 2
    # The commit input is user-controlled, so it reaches shell commands only via env.
    for job in jobs.values():
        assert not any("inputs." in step.get("run", "") for step in job["steps"])


def test_release_publishes_cd_build_before_tagging() -> None:
    jobs = workflow("release")["jobs"]
    steps = jobs["publish"]["steps"]
    names = [step["name"] for step in steps]
    sha = "${{ steps.commit.outputs.sha }}"
    find = names.index("Find CD build")
    download_build = names.index("Download package build")
    download_identity = names.index("Download deployment identity")
    verify = names.index("Verify CD build")
    validate = names.index("Validate unpublished version")
    check = names.index("Check build")
    publish = names.index("Publish to PyPI")
    assert find < download_build < verify < validate < check < publish
    assert download_identity < verify
    assert "Build package" not in names
    assert steps[find]["run"] == "make release-find-build"
    assert steps[find]["env"]["RELEASE_SHA"] == sha
    run_id = "${{ steps.build.outputs.run-id }}"
    assert jobs["publish"]["outputs"]["run-id"] == run_id
    assert steps[download_build]["with"]["name"] == f"dist-{sha}"
    assert steps[download_identity]["with"]["name"] == f"release-metadata-{sha}"
    assert steps[download_build]["with"]["path"] == "dist"
    for step in (steps[download_build], steps[download_identity]):
        assert step["with"]["run-id"] == run_id
        assert step["with"]["repository"] == "${{ github.repository }}"
    metadata_path = steps[download_identity]["with"]["path"] + "/release-metadata.json"
    assert steps[verify]["run"] == "make release-verify-build"
    assert steps[verify]["env"]["RELEASE_METADATA"] == metadata_path
    assert steps[validate]["run"] == "make validate-version-unpublished"
    assert steps[publish]["env"]["TWINE_PASSWORD"] == "${{ secrets.PYPI_API_TOKEN }}"

    steps = jobs["github-release"]["steps"]
    names = [step["name"] for step in steps]
    released = "${{ needs.publish.outputs.sha }}"
    assert steps[names.index("Check out release commit")]["with"]["ref"] == released
    download = steps[names.index("Download deployment identity")]
    assert download["with"]["run-id"] == "${{ needs.publish.outputs.run-id }}"
    assert download["with"]["name"] == f"release-metadata-{released}"
    release = steps[names.index("Publish GitHub release")]
    assert release["run"] == "make release"
    assert release["env"]["GH_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert release["env"]["RELEASE_METADATA"] == (
        download["with"]["path"] + "/release-metadata.json"
    )
    assert not any("pip install" in step.get("run", "") for step in steps)


@pytest.mark.parametrize("name", ["ci", "cd", "release"])
def test_workflow_job_environment_uses_available_contexts(name: str) -> None:
    """Job env is evaluated before runner and step contexts are available.

    GitHub's context-availability table allows these seven contexts at job env:
    https://docs.github.com/en/actions/reference/workflows-and-actions/contexts
    """
    available = {"github", "needs", "strategy", "matrix", "vars", "secrets", "inputs"}
    for job in workflow(name)["jobs"].values():
        for value in job.get("env", {}).values():
            for expression in re.findall(r"\$\{\{(.*?)\}\}", value):
                contexts = set(re.findall(r"(?<![\w.])([a-z_]\w*)\.", expression))
                assert contexts <= available, (
                    f"Unavailable contexts in {name} job env: {contexts - available}"
                )


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


def test_maintainer_release_instructions_reference_existing_workflows_and_targets() -> (
    None
):
    documentation = (ROOT / "CONTRIBUTING.md").read_text()
    section = documentation.split("## Release Process\n", 1)[1].split(
        "## Code Style\n", 1
    )[0]
    names = [workflow(name)["name"] for name in ("ci", "cd")]
    assert " → ".join(names) in section
    assert f"**{workflow('release')['name']}**" in section
    documented_commands = set(re.findall(r"make ([\w-]+)", section))
    targets = set(
        re.findall(r"^([\w-]+):", (ROOT / "Makefile").read_text(), re.MULTILINE)
    )
    assert documented_commands <= targets
    assert {
        "lint",
        "format-check",
        "type-check",
        "test",
        "validate-version-unpublished",
        "release-metadata",
        "release-resolve-commit",
        "release-find-build",
        "release-verify-build",
        "release",
    } <= documented_commands
