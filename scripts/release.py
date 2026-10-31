"""Prepare CD builds for release and publish GitHub releases without dependencies."""

# ruff: noqa: INP001
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

logger = logging.getLogger(__name__)


def record_metadata(output: Path) -> None:
    """Record the current checkout, checking the expected SHA when provided by CD."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    expected = os.environ.get("DEPLOYED_SHA")
    if expected and sha != expected:
        message = "Checkout does not match the successful CI commit"
        raise ValueError(message)
    with Path("pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    metadata = {"sha": sha, "name": project["name"], "version": project["version"]}
    output.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    logger.info(
        "Prepared deployment metadata for %s %s at %s in %s",
        metadata["name"],
        metadata["version"],
        sha,
        output,
    )


def read_metadata(path: Path) -> dict[str, str]:
    """Validate the deployment identity before using it in outputs or API calls."""
    data = json.loads(path.read_text(encoding="utf-8"))
    patterns = {
        "sha": r"[0-9a-f]{40}",
        "name": r"[A-Za-z0-9][A-Za-z0-9._-]*",
        "version": r"[A-Za-z0-9][A-Za-z0-9.!+_-]*",
    }
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), str) or not re.fullmatch(pattern, data[key])
        for key, pattern in patterns.items()
    ):
        message = "Invalid deployment metadata: expected sha, name, and version"
        raise ValueError(message)
    return {key: data[key] for key in patterns}


def release_notes(project: Path, version: str) -> str:
    """Extract exactly one nonempty curated section for the deployed version."""
    lines = (
        (project / "documentation" / "release-notes.md")
        .read_text(
            encoding="utf-8",
        )
        .splitlines()
    )
    matches = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^##\s+" + re.escape(version) + r"(?:\s|$)", line)
    ]
    if len(matches) != 1:
        message = f"Expected exactly one release-notes section for {version}"
        raise ValueError(message)
    start = matches[0] + 1
    end = next(
        (
            index
            for index in range(start, len(lines))
            if re.match(r"^##\s+", lines[index])
        ),
        len(lines),
    )
    notes = "\n".join(lines[start:end]).strip()
    if not notes:
        message = f"Empty release-notes section for {version}"
        raise ValueError(message)
    return notes


def validate_repository(repository: str) -> None:
    """Reject repository names that could alter GitHub API endpoints."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        message = "A valid owner/repository is required"
        raise ValueError(message)


def write_output(key: str, value: object) -> None:
    """Expose a step output to later workflow steps."""
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        stream.write(f"{key}={value}\n")


def api(endpoint: str) -> dict[str, Any] | None:
    """Read GitHub JSON, distinguishing absent resources from API failures."""
    result = subprocess.run(
        ["gh", "api", endpoint],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        if "(HTTP 404)" in result.stderr:
            return None
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            result.stdout,
            result.stderr,
        )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        message = f"Unexpected API response from {endpoint}"
        raise TypeError(message)
    return value


def tag_commit(repository: str, tag: str) -> str | None:
    """Resolve lightweight and annotated tags to their actual commit."""
    reference = api(f"repos/{repository}/git/ref/tags/{tag}")
    if reference is None:
        return None
    target = reference["object"]
    seen = set()
    while target["type"] == "tag":
        sha = target["sha"]
        if sha in seen:
            message = "Cycle in annotated release tag"
            raise ValueError(message)
        seen.add(sha)
        annotated = api(f"repos/{repository}/git/tags/{sha}")
        if annotated is None:
            message = "Annotated release tag is missing"
            raise ValueError(message)
        target = annotated["object"]
    if target["type"] != "commit" or not isinstance(target["sha"], str):
        message = "Release tag does not resolve to a commit"
        raise ValueError(message)
    return str(target["sha"])


def resolve_commit(commit: str, project: Path) -> None:
    """Expose the full SHA of a commit on the checked-out branch, defaulting to HEAD."""
    revision = commit.strip() or "HEAD"
    if revision != "HEAD":
        # Git accepts either case, and pasted SHAs are sometimes uppercase.
        revision = revision.lower()
    if revision != "HEAD" and not re.fullmatch(r"[0-9a-f]{7,40}", revision):
        message = "The release commit must be a hexadecimal commit SHA"
        raise ValueError(message)
    sha = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "--verify", f"{revision}^{{commit}}"],
        text=True,
    ).strip()
    result = subprocess.run(
        ["git", "-C", str(project), "merge-base", "--is-ancestor", sha, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 1:
        message = f"Release commit {sha} is not on the checked-out branch"
        raise ValueError(message)
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            result.stdout,
            result.stderr,
        )
    write_output("sha", sha)
    logger.info("Selected release commit %s", sha)


def is_cd_run(repository: str, run_id: int) -> bool:
    """Accept only master CD runs of this repository, not other uploaders."""
    run = api(f"repos/{repository}/actions/runs/{run_id}")
    return (
        run is not None
        and run.get("path") == ".github/workflows/cd.yml"
        and run.get("event") == "workflow_run"
        and run.get("head_branch") == "master"
        and run.get("repository", {}).get("full_name") == repository
    )


def find_build(repository: str, sha: str) -> None:
    """Expose the newest CD run that recorded a build for the commit."""
    validate_repository(repository)
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        message = "A full commit SHA is required"
        raise ValueError(message)
    listing = api(
        f"repos/{repository}/actions/artifacts?name=release-metadata-{sha}&per_page=100"
    )
    artifacts = [
        artifact
        for artifact in (listing or {}).get("artifacts", [])
        if not artifact["expired"] and artifact["workflow_run"]["head_sha"] == sha
    ]
    artifacts.sort(key=lambda artifact: artifact["created_at"], reverse=True)
    run_id = next(
        (
            int(artifact["workflow_run"]["id"])
            for artifact in artifacts
            if is_cd_run(repository, int(artifact["workflow_run"]["id"]))
        ),
        None,
    )
    if run_id is None:
        message = (
            f"No unexpired CD build for {sha}; wait for CD on master to succeed, "
            "rerun it if it failed, or release a newer master commit if its "
            "artifacts expired"
        )
        raise ValueError(message)
    write_output("run-id", run_id)
    logger.info("Selected CD run %s for %s", run_id, sha)


def verify_checkout(metadata: Path, project: Path) -> dict[str, str]:
    """Require the CD build identity to match the checkout being released."""
    identity = read_metadata(metadata)
    sha = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    with (project / "pyproject.toml").open("rb") as stream:
        package = tomllib.load(stream)["project"]
    if sha != identity["sha"] or any(
        package[key] != identity[key] for key in ("name", "version")
    ):
        message = "Deployment metadata does not match the release checkout"
        raise ValueError(message)
    logger.info(
        "CD build of %s %s matches checkout %s",
        identity["name"],
        identity["version"],
        sha,
    )
    return identity


def publish_release(metadata: Path, project: Path, repository: str) -> None:
    """Publish a release for the deployed checkout, preserving existing releases."""
    validate_repository(repository)
    identity = verify_checkout(metadata, project)
    sha = identity["sha"]
    tag = f"v{identity['version']}"
    notes = "## What's Changed\n\n" + release_notes(project, identity["version"])
    notes += (
        f"\n\n[View on PyPI](https://pypi.org/project/"
        f"{identity['name']}/{identity['version']}/)\n"
    )
    logger.info("Preparing release %s at deployed commit %s", tag, sha)
    existing = api(f"repos/{repository}/releases/tags/{tag}")
    target = tag_commit(repository, tag)
    if target is not None and target != sha:
        message = f"Existing tag {tag} points to a different commit"
        raise ValueError(message)
    if existing is not None:
        if existing["draft"] or target != sha:
            message = f"Existing release {tag} is a draft or has no matching tag"
            raise ValueError(message)
        url = str(existing["html_url"])
        logger.info("Release %s already published at %s; skipping", tag, sha)
        outcome = "Already published"
    else:
        with TemporaryDirectory(prefix="release-notes-") as temporary:
            path = Path(temporary) / "notes.md"
            path.write_text(notes, encoding="utf-8")
            command = [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                repository,
                "--title",
                tag,
                "--target",
                sha,
                "--notes-file",
                str(path),
            ]
            if target is not None:
                command.append("--verify-tag")
            url = subprocess.check_output(command, text=True).strip()
        logger.info("Published %s", url)
        outcome = "Published"
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(f"## GitHub Release\n\n{outcome}: [{tag}]({url})\n")


def main() -> int:
    """Run release tooling, logging failures and returning a nonzero status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    metadata = commands.add_parser("metadata", help="Record deployment identity")
    metadata.add_argument("--output", type=Path, default=Path("release-metadata.json"))
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    resolve = commands.add_parser(
        "resolve-commit", help="Select a commit on the checked-out branch"
    )
    resolve.add_argument("--commit", default="")
    resolve.add_argument("--project", type=Path, default=Path())
    find = commands.add_parser("find-build", help="Select the CD run for a commit")
    find.add_argument("--sha", required=True)
    find.add_argument("--repository", default=repository)
    for name in ("verify-build", "publish"):
        command = commands.add_parser(name)
        command.add_argument(
            "--metadata", type=Path, default=Path("release-metadata.json")
        )
        command.add_argument("--project", type=Path, default=Path())
        if name == "publish":
            command.add_argument("--repository", default=repository)
    args = parser.parse_args()
    try:
        if args.command == "metadata":
            record_metadata(args.output)
        elif args.command == "resolve-commit":
            resolve_commit(args.commit, args.project)
        elif args.command == "find-build":
            find_build(args.repository, args.sha)
        elif args.command == "verify-build":
            verify_checkout(args.metadata, args.project)
        else:
            publish_release(args.metadata, args.project, args.repository)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        logger.exception("Release tooling failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
