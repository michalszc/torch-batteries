"""Prepare deployment metadata and GitHub releases without package dependencies."""

# ruff: noqa: INP001
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tomllib
from pathlib import Path

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


def main() -> int:
    """Run release tooling, logging failures and returning a nonzero status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    metadata = commands.add_parser("metadata", help="Record deployment identity")
    metadata.add_argument("--output", type=Path, default=Path("release-metadata.json"))
    args = parser.parse_args()
    try:
        record_metadata(args.output)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        logger.exception("Release tooling failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
