"""Architecture checks for focused production modules."""

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "torch_batteries"


def _production_modules() -> list[Path]:
    """Return every production Python module."""
    return sorted(SOURCE_ROOT.rglob("*.py"))


@pytest.mark.parametrize("path", _production_modules(), ids=str)
def test_module_defines_at_most_one_class_and_no_peer_functions(path: Path) -> None:
    """Keep each class-like declaration in a focused source module.

    Args:
        path: Production module inspected by the architecture rule.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    functions = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]

    assert len(classes) <= 1, f"{path} defines multiple classes: {classes}"
    assert not (classes and functions), (
        f"{path} mixes class {classes[0]} with functions: {functions}"
    )
    if path.name == "__init__.py":
        assert not classes, f"Package initializer {path} must contain exports only."
        assert not functions, f"Package initializer {path} must contain exports only."
