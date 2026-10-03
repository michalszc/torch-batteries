"""Tests for torch_batteries package imports and basic functionality."""

import builtins
import importlib
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import torch
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
from setuptools.config.pyprojecttoml import (  # type: ignore[import-untyped]
    read_configuration,
)
from setuptools.discovery import PEP420PackageFinder  # type: ignore[import-untyped]
from torch import nn

import torch_batteries
from torch_batteries import data, events, trainer, utils
from torch_batteries.events import Event, EventHandler, charge
from torch_batteries.trainer import Battery
from torch_batteries.utils import batch, device, logging, progress

PROJECT_ROOT = Path(__file__).parents[3]
logger = logging.get_logger("tests.contracts.package")


@pytest.fixture
def project_configuration() -> dict[str, Any]:
    """Read source configuration rather than stale installed distribution metadata."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as configuration:
        return tomllib.load(configuration)


def test_project_version_matches_runtime(project_configuration: dict[str, Any]) -> None:
    """Keep the public runtime and distribution version in agreement."""
    assert project_configuration["project"]["version"] == torch_batteries.__version__


def test_dependencies_and_extras(project_configuration: dict[str, Any]) -> None:
    """Declare direct imports while keeping integrations optional."""
    project = project_configuration["project"]
    dependencies: dict[str, Requirement] = {
        canonicalize_name(requirement.name): requirement
        for value in project["dependencies"]
        for requirement in [Requirement(value)]
    }
    assert {"torch", "tqdm", "pyyaml", "typing-extensions"} <= dependencies.keys()
    assert "wandb" not in dependencies
    assert "numpy" not in dependencies
    assert Version("4.10.0") in dependencies["typing-extensions"].specifier

    extras = project["optional-dependencies"]
    assert {"example", "wandb", "all"} <= extras.keys()
    all_requirements = [Requirement(value) for value in extras["all"]]
    assert len(all_requirements) == 1
    combined = all_requirements[0]
    assert canonicalize_name(combined.name) == canonicalize_name(project["name"])
    assert combined.extras == {"example", "wandb"}
    assert combined.marker is None
    assert combined.url is None
    assert not combined.specifier
    for extra in combined.extras:
        assert extras[extra]
        for value in extras[extra]:
            assert canonicalize_name(Requirement(value).name) != canonicalize_name(
                project["name"],
            )


def test_backend_supports_license_metadata(
    project_configuration: dict[str, Any],
) -> None:
    """Require a backend supporting SPDX expressions and included license files."""
    backend = project_configuration["build-system"]
    assert backend["build-backend"] == "setuptools.build_meta"
    requirements: dict[str, Requirement] = {
        canonicalize_name(requirement.name): requirement
        for value in backend["requires"]
        for requirement in [Requirement(value)]
    }
    assert "wheel" in requirements
    assert Version("77.0.3") in requirements["setuptools"].specifier
    assert Version("77.0.2") not in requirements["setuptools"].specifier
    project = project_configuration["project"]
    assert project["license"] == "Apache-2.0"
    assert {"LICENSE", "NOTICE"} <= set(project["license-files"])
    for filename in project["license-files"]:
        assert (PROJECT_ROOT / filename).is_file()


def test_package_discovery_excludes_unrelated_packages(
    project_configuration: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Include real subpackages while excluding unrelated and similarly named ones."""
    discovery = project_configuration["tool"]["setuptools"]["packages"]["find"]
    assert discovery["where"] == ["src"]
    for name in (
        "torch_batteries",
        "torch_batteries.data",
        "torch_batteries_extra",
        "tests",
    ):
        package = tmp_path.joinpath(*name.split("."))
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()

    found = PEP420PackageFinder.find(str(tmp_path), include=discovery["include"])
    logger.debug("Discovered fixture packages: %s", found)
    assert set(found) == {"torch_batteries", "torch_batteries.data"}
    actual = PEP420PackageFinder.find(
        str(PROJECT_ROOT / "src"),
        include=discovery["include"],
    )
    expected = {
        ".".join(path.parent.relative_to(PROJECT_ROOT / "src").parts)
        for path in (PROJECT_ROOT / "src" / "torch_batteries").rglob("__init__.py")
    }
    assert set(actual) == expected


def test_setuptools_configuration_includes_typing_marker() -> None:
    """Validate backend configuration and marker inclusion without a build."""
    configuration = read_configuration(str(PROJECT_ROOT / "pyproject.toml"))
    package_data = configuration["tool"]["setuptools"]["package-data"]
    assert "py.typed" in package_data["torch_batteries"]
    assert (PROJECT_ROOT / "src" / "torch_batteries" / "py.typed").is_file()


def test_package_import() -> None:
    """Test that the main package can be imported."""
    assert torch_batteries is not None
    assert hasattr(torch_batteries, "__version__")


def test_submodules_import() -> None:
    """Test that all submodules can be imported."""
    assert events is not None
    assert data is not None
    assert trainer is not None
    assert utils is not None


def test_core_classes_import() -> None:
    """Test that core classes can be imported."""
    assert Event is not None
    assert EventHandler is not None
    assert charge is not None
    assert Battery is not None
    assert torch_batteries.DataPack is not None
    assert torch_batteries.DataPackHandler is not None
    assert torch_batteries.DatasetBundle is not None
    assert torch_batteries.DataLoaderConfig is not None
    assert torch_batteries.DataContext is not None


def test_utils_import() -> None:
    """Test that utility modules can be imported."""
    assert batch is not None
    assert device is not None
    assert logging is not None
    assert progress is not None


def test_basic_model_creation() -> None:
    """Test basic model and battery creation."""
    model = nn.Linear(10, 1)
    battery = Battery(model)

    assert battery.model is model
    assert isinstance(battery.device, torch.device)
    assert battery.optimizer is None


def test_plain_install_imports_without_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test package imports when the optional wandb dependency is unavailable."""
    real_import = builtins.__import__

    def block_wandb_import(name: str, *args: Any, **kwargs: Any) -> Any:
        level = kwargs.get(
            "level",
            args[-1] if args and isinstance(args[-1], int) else 0,
        )
        if level == 0 and (name == "wandb" or name.startswith("wandb.")):
            msg = "No module named 'wandb'"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monkeypatch.setattr(builtins, "__import__", block_wandb_import)

    tracking_module = importlib.import_module("torch_batteries.tracking")
    wandb_module = importlib.import_module("torch_batteries.tracking.wandb")

    importlib.reload(wandb_module)
    importlib.reload(tracking_module)
    reloaded_package = importlib.reload(torch_batteries)

    imported_battery = reloaded_package.Battery
    run = reloaded_package.Run
    wandb_tracker = tracking_module.WandbTracker

    assert imported_battery is Battery
    assert run is not None
    assert wandb_tracker is not None

    with pytest.raises(ImportError, match=r"pip install torch-batteries\[wandb\]"):
        wandb_tracker(project="test-project")
