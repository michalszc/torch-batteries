"""Protect the documented public API against accidental refactors."""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from tests.contracts.public_api.manifest import (
    DEPRECATED_PARAMETERS,
    PUBLIC_EXPORTS,
    PUBLIC_MEMBERS,
    PUBLIC_PARAMETERS,
    REFERENCE_EXCLUSIONS,
    REFERENCE_PAGES,
)


def _resolve(path: str) -> Any:
    """Resolve a module, class, or member from its public dotted path."""
    parts = path.split(".")
    for index in range(len(parts), 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:index]))
        except ModuleNotFoundError:
            continue
        for part in parts[index:]:
            value = getattr(value, part)
        return value
    raise ModuleNotFoundError(path)


def test_protected_exports_remain_public() -> None:
    """Every protected export remains available through its documented package."""
    for module_name, expected in PUBLIC_EXPORTS.items():
        module = importlib.import_module(module_name)
        assert expected <= set(module.__all__)
        for name in expected:
            assert getattr(module, name) is not None


def test_protected_members_remain_public() -> None:
    """Every documented method and attribute remains on its public owner."""
    for owner_name, members in PUBLIC_MEMBERS.items():
        owner = _resolve(owner_name)
        for member in members:
            assert hasattr(owner, member), f"{owner_name}.{member} is missing"


def test_protected_parameter_defaults_remain_compatible() -> None:
    """Existing parameters retain their defaults while additions remain allowed."""
    for callable_name, expected in PUBLIC_PARAMETERS.items():
        parameters = inspect.signature(_resolve(callable_name)).parameters
        for name, default in expected.items():
            assert name in parameters, f"{callable_name} lost parameter {name}"
            assert parameters[name].default == default


def test_deprecated_parameters_are_explicitly_tracked() -> None:
    """Deprecated compatibility remains protected until its removal release."""
    for callable_name, expected in DEPRECATED_PARAMETERS.items():
        parameters = inspect.signature(_resolve(callable_name)).parameters
        assert expected <= set(parameters)


def test_protected_endpoints_remain_in_generated_reference(tmp_path: Path) -> None:
    """Every protected endpoint remains addressable in generated API HTML."""
    environment = os.environ.copy()
    environment["NO_MKDOCS_2_WARNING"] = "1"
    subprocess.run(
        ["mkdocs", "build", "--strict", "--site-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    expected_paths: set[str] = set()
    for module_name, exports in PUBLIC_EXPORTS.items():
        if module_name in REFERENCE_PAGES:
            expected_paths.update(f"{module_name}.{name}" for name in exports)
    for owner_name, members in PUBLIC_MEMBERS.items():
        expected_paths.add(owner_name)
        expected_paths.update(f"{owner_name}.{member}" for member in members)

    missing: list[str] = []
    for path in sorted(expected_paths - REFERENCE_EXCLUSIONS):
        page_name = next(
            page
            for module_name, page in REFERENCE_PAGES.items()
            if path == module_name or path.startswith(f"{module_name}.")
        )
        html = (tmp_path / page_name).read_text(encoding="utf-8")
        if f'id="{path}"' not in html:
            missing.append(path)

    assert not missing, f"Missing documented API endpoints: {missing}"
