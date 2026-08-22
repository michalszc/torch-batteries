"""Enforce baseline documentation coverage for the public Python API."""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "torch_batteries"


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function node is decorated as an overload."""
    return any(
        (isinstance(decorator, ast.Name) and decorator.id == "overload")
        or (isinstance(decorator, ast.Attribute) and decorator.attr == "overload")
        for decorator in node.decorator_list
    )


def _is_property_setter(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a method is a property setter or deleter."""
    return any(
        isinstance(decorator, ast.Attribute) and decorator.attr in {"setter", "deleter"}
        for decorator in node.decorator_list
    )


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return user-facing parameter names declared by a callable."""
    parameters = [
        argument.arg
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        if argument.arg not in {"self", "cls"}
    ]
    if node.args.vararg is not None:
        parameters.append(node.args.vararg.arg)
    if node.args.kwarg is not None:
        parameters.append(node.args.kwarg.arg)
    return parameters


def _public_callables() -> Iterator[
    tuple[Path, str, ast.FunctionDef | ast.AsyncFunctionDef, str | None]
]:
    """Yield public callables with the docstring that describes their parameters."""
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and not _is_overload(node):
                    yield path, node.name, node, ast.get_docstring(node)
                continue
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            class_docstring = ast.get_docstring(node)
            for method in node.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if method.name.startswith("_") and method.name != "__init__":
                    continue
                if _is_overload(method) or _is_property_setter(method):
                    continue
                docstring = (
                    ast.get_docstring(method) or class_docstring
                    if method.name == "__init__"
                    else ast.get_docstring(method)
                )
                yield path, f"{node.name}.{method.name}", method, docstring


def _public_classes() -> Iterator[tuple[Path, ast.ClassDef]]:
    """Yield every source class whose name is part of a public module surface."""
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                yield path, node


@pytest.mark.parametrize(("path", "node"), list(_public_classes()))
def test_public_class_has_docstring(path: Path, node: ast.ClassDef) -> None:
    """Every public class provides a rendered API description."""
    location = f"{path.relative_to(SOURCE_ROOT.parent.parent)}:{node.lineno}"
    assert ast.get_docstring(node), f"{location} {node.name} has no docstring"


@pytest.mark.parametrize(
    ("path", "qualified_name", "node", "docstring"),
    list(_public_callables()),
    ids=lambda value: str(value) if isinstance(value, (Path, str)) else None,
)
def test_public_callable_documents_parameters(
    path: Path,
    qualified_name: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    docstring: str | None,
) -> None:
    """Public callables have docstrings that name every accepted parameter."""
    location = f"{path.relative_to(SOURCE_ROOT.parent.parent)}:{node.lineno}"
    assert docstring, f"{location} {qualified_name} has no docstring"

    missing = [
        parameter
        for parameter in _parameters(node)
        if re.search(rf"(?m)^\s*\**{re.escape(parameter)}:", docstring) is None
    ]
    assert not missing, (
        f"{location} {qualified_name} does not document parameters: "
        f"{', '.join(missing)}"
    )
