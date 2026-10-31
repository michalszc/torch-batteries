"""Offline tests of CD build selection, release publication, and retry safety."""

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
        "artifacts": None,
        "runs": {},
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
        elif "/actions/artifacts" in endpoint:
            value = state["artifacts"]
        elif "/actions/runs/" in endpoint:
            value = state["runs"].get(int(endpoint.rsplit("/", 1)[1]))
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
def test_invalid_metadata_fails_verification(
    project: Path,
    github: dict[str, Any],
    identity: Any,
) -> None:
    (project / "metadata.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="Invalid deployment metadata"):
        TOOL.verify_checkout(project / "metadata.json", project)


def test_verify_checkout_accepts_matching_build(
    project: Path, github: dict[str, Any]
) -> None:
    identity = TOOL.verify_checkout(project / "metadata.json", project)
    assert identity == {"sha": SHA, "name": "torch-batteries", "version": "1.0.0"}
    assert not github["commands"]


def artifact(run_id: int, created_at: str, *, expired: bool = False) -> dict[str, Any]:
    return {
        "expired": expired,
        "created_at": created_at,
        "workflow_run": {"id": run_id, "head_sha": SHA},
    }


def cd_run(**overrides: Any) -> dict[str, Any]:
    return {
        "path": ".github/workflows/cd.yml",
        "event": "workflow_run",
        "head_branch": "master",
        "repository": {"full_name": "owner/repo"},
        **overrides,
    }


def test_find_build_selects_newest_unexpired_cd_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, github: dict[str, Any]
) -> None:
    github["runs"] = {1: cd_run(), 2: cd_run(), 3: cd_run()}
    github["artifacts"] = {
        "artifacts": [
            artifact(1, "2026-01-01T00:00:00Z"),
            artifact(3, "2026-01-03T00:00:00Z", expired=True),
            artifact(2, "2026-01-02T00:00:00Z"),
        ]
    }
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    TOOL.find_build("owner/repo", SHA)
    assert output.read_text() == "run-id=2\n"
    endpoint = github["commands"][0][2]
    assert endpoint.startswith("repos/owner/repo/actions/artifacts?")
    assert f"name=release-metadata-{SHA}" in endpoint


@pytest.mark.parametrize(
    "run",
    [
        cd_run(path=".github/workflows/ci.yml"),
        cd_run(event="workflow_dispatch"),
        cd_run(head_branch="feature"),
        cd_run(repository={"full_name": "fork/repo"}),
    ],
)
def test_find_build_skips_runs_outside_master_cd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    github: dict[str, Any],
    run: dict[str, Any],
) -> None:
    github["runs"] = {1: cd_run(), 2: run}
    github["artifacts"] = {
        "artifacts": [
            artifact(1, "2026-01-01T00:00:00Z"),
            artifact(2, "2026-01-02T00:00:00Z"),
        ]
    }
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    TOOL.find_build("owner/repo", SHA)
    assert output.read_text() == "run-id=1\n"


def test_find_build_requires_cd_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, github: dict[str, Any]
) -> None:
    github["runs"] = {1: cd_run(path=".github/workflows/ci.yml")}
    github["artifacts"] = {"artifacts": [artifact(1, "2026-01-01T00:00:00Z")]}
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    with pytest.raises(ValueError, match="No unexpired CD build"):
        TOOL.find_build("owner/repo", SHA)
    assert not output.exists()


@pytest.mark.parametrize(
    "artifacts",
    [
        {"artifacts": []},
        {"artifacts": [artifact(1, "2026-01-01T00:00:00Z", expired=True)]},
    ],
)
def test_find_build_requires_unexpired_cd_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    github: dict[str, Any],
    artifacts: dict[str, Any],
) -> None:
    github["artifacts"] = artifacts
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    with pytest.raises(ValueError, match="No unexpired CD build"):
        TOOL.find_build("owner/repo", SHA)
    assert not output.exists()


@pytest.mark.parametrize(
    ("repository", "sha", "error"),
    [
        ("owner", SHA, "owner/repository"),
        ("owner/repo", "master", "full commit SHA"),
        ("owner/repo", SHA[:7], "full commit SHA"),
    ],
)
def test_find_build_validates_inputs_before_api(
    github: dict[str, Any], repository: str, sha: str, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        TOOL.find_build(repository, sha)
    assert not github["commands"]


@pytest.fixture
def history(tmp_path: Path) -> dict[str, Any]:
    """Create a real repository with master history and an unmerged side commit."""
    repository = tmp_path / "history"
    repository.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments], text=True
        ).strip()

    git("init", "--quiet")
    git("symbolic-ref", "HEAD", "refs/heads/master")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("commit", "--quiet", "--allow-empty", "-m", "bump")
    bump = git("rev-parse", "HEAD")
    git("commit", "--quiet", "--allow-empty", "-m", "docs")
    head = git("rev-parse", "HEAD")
    git("checkout", "--quiet", "-b", "side", bump)
    git("commit", "--quiet", "--allow-empty", "-m", "side")
    side = git("rev-parse", "HEAD")
    git("checkout", "--quiet", "master")
    return {"path": repository, "bump": bump, "head": head, "side": side}


@pytest.mark.parametrize(("commit", "expected"), [("", "head"), ("  ", "head")])
def test_resolve_commit_defaults_to_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    history: dict[str, Any],
    commit: str,
    expected: str,
) -> None:
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    TOOL.resolve_commit(commit, history["path"])
    assert output.read_text() == f"sha={history[expected]}\n"


@pytest.mark.parametrize("length", [7, 40])
@pytest.mark.parametrize("upper", [False, True])
def test_resolve_commit_accepts_earlier_branch_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    history: dict[str, Any],
    length: int,
    *,
    upper: bool,
) -> None:
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    commit = history["bump"][:length]
    TOOL.resolve_commit(commit.upper() if upper else commit, history["path"])
    assert output.read_text() == f"sha={history['bump']}\n"


def test_resolve_commit_rejects_commit_outside_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history: dict[str, Any]
) -> None:
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    with pytest.raises(ValueError, match="not on the checked-out branch"):
        TOOL.resolve_commit(history["side"], history["path"])
    assert not output.exists()


@pytest.mark.parametrize("commit", ["master", "HEAD~1", "abc", "a" * 41, "$(id)"])
def test_resolve_commit_rejects_non_sha_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    history: dict[str, Any],
    commit: str,
) -> None:
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    with pytest.raises(ValueError, match="hexadecimal commit SHA"):
        TOOL.resolve_commit(commit, history["path"])
    assert not output.exists()


def test_resolve_commit_rejects_unknown_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history: dict[str, Any]
) -> None:
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    with pytest.raises(subprocess.CalledProcessError):
        TOOL.resolve_commit("f" * 40, history["path"])


def test_cli_logs_failure(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["release.py", "verify-build", "--metadata", str(project / "missing.json")],
    )
    assert TOOL.main() == 1
    assert "Release tooling failed" in caplog.text
