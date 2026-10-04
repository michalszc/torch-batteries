"""Offline tests of release publication, identity validation, and retry safety."""

from __future__ import annotations

import json
import subprocess
from importlib import util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[3]
SHA = "a" * 40
SPEC = util.spec_from_file_location("release_tooling", ROOT / "scripts" / "release.py")
assert SPEC is not None
assert SPEC.loader is not None
TOOL = util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "torch-batteries"\nversion = "1.0.0"\n',
    )
    documentation = tmp_path / "documentation"
    documentation.mkdir()
    (documentation / "release-notes.md").write_text(
        "# Release Notes\n\n## 1.0.0 — Today\n\n- Current change.\n\n"
        "## 0.9.0 — Earlier\n\n- Old change.\n",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "sha": SHA,
                "name": "torch-batteries",
                "version": "1.0.0",
            }
        )
    )
    return tmp_path


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Replace every GitHub and git invocation; never access network or publish."""
    state: dict[str, Any] = {
        "release": None,
        "ref": None,
        "annotated": None,
        "api_error": None,
        "create_error": False,
        "commands": [],
    }

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[:2] == ["gh", "api"]
        state["commands"].append(command)
        if state["api_error"]:
            return subprocess.CompletedProcess(command, 1, "", state["api_error"])
        endpoint = command[2]
        if "/releases/tags/" in endpoint:
            value = state["release"]
        elif "/git/ref/" in endpoint:
            value = state["ref"]
        else:
            value = state["annotated"]
        if value is None:
            return subprocess.CompletedProcess(
                command, 1, "", "gh: Not Found (HTTP 404)"
            )
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    def output(command: list[str], **kwargs: Any) -> str:
        if command[0] == "git":
            return SHA + "\n"
        assert command[:3] == ["gh", "release", "create"]
        state["commands"].append(command)
        if state["create_error"]:
            raise subprocess.CalledProcessError(1, command)
        state["notes"] = Path(command[command.index("--notes-file") + 1]).read_text()
        return "https://github.com/owner/repo/releases/tag/v1.0.0\n"

    monkeypatch.setattr(TOOL.subprocess, "run", run)
    monkeypatch.setattr(TOOL.subprocess, "check_output", output)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    return state


def publish(project: Path) -> None:
    TOOL.publish_release(project / "metadata.json", project, "owner/repo")


def test_creates_release_at_deployed_commit(
    project: Path, github: dict[str, Any]
) -> None:
    publish(project)
    create = github["commands"][-1]
    assert create[:4] == ["gh", "release", "create", "v1.0.0"]
    assert create[create.index("--target") + 1] == SHA
    assert create[create.index("--title") + 1] == "v1.0.0"
    assert create[create.index("--repo") + 1] == "owner/repo"
    assert "--verify-tag" not in create
    assert github["notes"].startswith("## What's Changed\n\n- Current change.")
    assert "Old change." not in github["notes"]
    assert "https://pypi.org/project/torch-batteries/1.0.0/" in github["notes"]
    assert "Published: [v1.0.0]" in (project / "summary.md").read_text()


def test_existing_tag_is_verified(project: Path, github: dict[str, Any]) -> None:
    github["ref"] = {"object": {"type": "commit", "sha": SHA}}
    publish(project)
    assert "--verify-tag" in github["commands"][-1]


def test_resolves_annotated_tag(project: Path, github: dict[str, Any]) -> None:
    github["ref"] = {"object": {"type": "tag", "sha": "b" * 40}}
    github["annotated"] = {"object": {"type": "commit", "sha": SHA}}
    publish(project)
    assert "--verify-tag" in github["commands"][-1]


def test_duplicate_release_skips_creation(
    project: Path, github: dict[str, Any]
) -> None:
    github["release"] = {
        "draft": False,
        "html_url": "https://github.com/owner/repo/releases/tag/v1.0.0",
    }
    github["ref"] = {"object": {"type": "commit", "sha": SHA}}
    publish(project)
    assert all(command[1] == "api" for command in github["commands"])
    assert "Already published" in (project / "summary.md").read_text()


@pytest.mark.parametrize("existing_release", [False, True])
def test_conflicting_tag_fails(
    project: Path,
    github: dict[str, Any],
    *,
    existing_release: bool,
) -> None:
    github["ref"] = {"object": {"type": "commit", "sha": "b" * 40}}
    if existing_release:
        github["release"] = {"draft": False}
    with pytest.raises(ValueError, match="different commit"):
        publish(project)
    assert all(command[1] == "api" for command in github["commands"])


def test_draft_release_fails(project: Path, github: dict[str, Any]) -> None:
    github["release"] = {"draft": True}
    github["ref"] = {"object": {"type": "commit", "sha": SHA}}
    with pytest.raises(ValueError, match="draft"):
        publish(project)


@pytest.mark.parametrize(
    "error", ["Forbidden (HTTP 403)", "Server error (HTTP 500)", "connection refused"]
)
def test_api_failure_does_not_create_release(
    project: Path,
    github: dict[str, Any],
    error: str,
) -> None:
    github["api_error"] = error
    with pytest.raises(subprocess.CalledProcessError):
        publish(project)
    assert all(command[1] == "api" for command in github["commands"])


def test_creation_failure_has_no_success_summary(
    project: Path, github: dict[str, Any]
) -> None:
    github["create_error"] = True
    with pytest.raises(subprocess.CalledProcessError):
        publish(project)
    assert not (project / "summary.md").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [("sha", "b" * 40), ("version", "2.0.0"), ("name", "other-package")],
)
def test_mismatched_metadata_fails_before_api(
    project: Path,
    github: dict[str, Any],
    key: str,
    value: str,
) -> None:
    path = project / "metadata.json"
    identity = json.loads(path.read_text())
    identity[key] = value
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="does not match"):
        publish(project)
    assert not github["commands"]


def test_missing_metadata_fails(project: Path, github: dict[str, Any]) -> None:
    with pytest.raises(FileNotFoundError):
        TOOL.publish_release(project / "missing.json", project, "owner/repo")
    assert not github["commands"]


@pytest.mark.parametrize(
    "content",
    [
        "## 0.9.0\n- Old.\n",
        "## 1.0.0\n- One.\n## 1.0.0\n- Two.\n",
        "## 1.0.0\n\n## 0.9.0\n- Old.\n",
    ],
)
def test_invalid_notes_fail_before_api(
    project: Path,
    github: dict[str, Any],
    content: str,
) -> None:
    (project / "documentation" / "release-notes.md").write_text(content)
    with pytest.raises(ValueError, match="release-notes section"):
        publish(project)
    assert not github["commands"]


@pytest.mark.parametrize(
    "identity",
    [
        [],
        {},
        {"sha": "master", "name": "pkg", "version": "1.0"},
        {"sha": SHA, "name": "pkg", "version": "1.0\nsha=master"},
    ],
)
def test_invalid_metadata_cannot_select_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: Any,
) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(identity))
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    with pytest.raises(ValueError, match="Invalid deployment metadata"):
        TOOL.select_deployed_commit(metadata)
    assert not output.exists()


def test_select_deployed_commit_uses_metadata_sha(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = project / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    TOOL.select_deployed_commit(project / "metadata.json")
    assert output.read_text() == f"sha={SHA}\n"


def test_cli_logs_failure(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["release.py", "select-commit", "--metadata", str(project / "missing.json")],
    )
    assert TOOL.main() == 1
    assert "Release tooling failed" in caplog.text
