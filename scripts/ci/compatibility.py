"""Portable environment and installation commands for the compatibility matrix."""

import argparse
import json
import platform
import subprocess
import sys
import tomllib
from importlib.metadata import distributions
from pathlib import Path


def install_test_dependencies(project_root: Path) -> None:
    """Install declared test/build requirements without depending on pip --group."""
    with (project_root / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    requirements = [
        *config["dependency-groups"]["test"],
        *config["build-system"]["requires"],
        "build>=1.0.0",
        "packaging",
        "numpy",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", *requirements], check=True)


def install_wheel(directory: Path) -> None:
    """Install exactly one built wheel while retaining the selected CPU PyTorch."""
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        message = f"Expected exactly one wheel in {directory}; found {len(wheels)}"
        raise ValueError(message)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            str(wheels[0]),
        ],
        check=True,
    )


def record_environment(destination: Path) -> None:
    """Record interpreter, native architecture, and installed dependency versions."""
    details = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {item.metadata["Name"]: item.version for item in distributions()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(details, indent=2))


def main() -> None:
    """Run one compatibility setup/reporting operation in the current checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["install-test-dependencies", "install-wheel", "record-environment"],
    )
    action = parser.parse_args().action
    if action == "install-test-dependencies":
        install_test_dependencies(Path.cwd())
    elif action == "install-wheel":
        install_wheel(Path("dist/compatibility"))
    else:
        record_environment(Path("compatibility-results/environment.json"))


if __name__ == "__main__":
    main()
