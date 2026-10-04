"""Prepare deployment metadata and GitHub releases without package dependencies."""

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


def select_deployed_commit(metadata: Path) -> None:
    """Expose only a validated full commit SHA to the workflow checkout step."""
    sha = read_metadata(metadata)["sha"]
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        stream.write(f"sha={sha}\n")
    logger.info("Selected deployed commit %s", sha)


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


def publish_release(metadata: Path, project: Path, repository: str) -> None:
    """Publish a release for the deployed checkout, preserving existing releases."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        message = "A valid owner/repository is required"
        raise ValueError(message)
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
    for name in ("select-commit", "publish"):
        command = commands.add_parser(name)
        command.add_argument(
            "--metadata", type=Path, default=Path("release-metadata.json")
        )
        if name == "publish":
            command.add_argument("--project", type=Path, default=Path())
            command.add_argument(
                "--repository", default=os.environ.get("GITHUB_REPOSITORY", "")
            )
    args = parser.parse_args()
    try:
        if args.command == "metadata":
            record_metadata(args.output)
        elif args.command == "select-commit":
            select_deployed_commit(args.metadata)
        else:
            publish_release(args.metadata, args.project, args.repository)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        logger.exception("Release tooling failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
