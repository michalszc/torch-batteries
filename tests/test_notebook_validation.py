"""Contract tests for metadata-free static notebook validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook, new_output

PROJECT_ROOT = Path(__file__).parents[1]
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_notebooks.sh"
VERSION = "0.8.0"


@pytest.fixture
def validation_project(tmp_path: Path) -> Path:
    """Create the minimum importable project used by validator subprocesses."""
    package = tmp_path / "src" / "torch_batteries"
    package.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "fixture"\nversion = "{VERSION}"\n',
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        f'__version__ = "{VERSION}"\n',
        encoding="utf-8",
    )
    return tmp_path


def notebook(*, version_cells: int = 1) -> nbformat.NotebookNode:
    """Build a small, fully executed notebook."""
    cells = []
    for execution_count in range(1, version_cells + 1):
        cells.append(
            new_code_cell(
                "import torch_batteries\nprint(torch_batteries.__version__)",
                execution_count=execution_count,
                outputs=[
                    new_output(
                        "stream",
                        name="stdout",
                        text=f"torch-batteries version: {VERSION}\n",
                    )
                ],
            )
        )
    cells.append(new_code_cell("answer = 42", execution_count=version_cells + 1))
    return cast("nbformat.NotebookNode", new_notebook(cells=cells))


def write_notebook(path: Path, value: nbformat.NotebookNode) -> None:
    """Write a fixture notebook in canonical nbformat form."""
    nbformat.write(value, path)


def validate(project: Path, notebook_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the Bash validator against one fixture notebook."""
    return subprocess.run(
        ["bash", str(VALIDATOR), "--project-root", str(project), str(notebook_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_accepts_valid_executed_notebook(
    validation_project: Path, tmp_path: Path
) -> None:
    path = tmp_path / "valid.ipynb"
    write_notebook(path, notebook())

    result = validate(validation_project, path)

    assert result.returncode == 0
    assert f"executed with torch-batteries {VERSION}" in result.stdout


def test_accepts_unexecuted_installation_cell(
    validation_project: Path, tmp_path: Path
) -> None:
    value = notebook()
    value.cells.insert(
        0,
        new_code_cell(
            '%pip install "torch-batteries[example]"',
            execution_count=None,
            outputs=[],
        ),
    )
    path = tmp_path / "installation.ipynb"
    write_notebook(path, value)

    result = validate(validation_project, path)

    assert result.returncode == 0


def test_rejects_saved_installation_output(
    validation_project: Path, tmp_path: Path
) -> None:
    value = notebook()
    value.cells.insert(
        0,
        new_code_cell(
            '%pip install "torch-batteries[example]"',
            execution_count=1,
            outputs=[
                new_output("stream", name="stdout", text="Successfully installed")
            ],
        ),
    )
    path = tmp_path / "installation-output.ipynb"
    write_notebook(path, value)

    result = validate(validation_project, path)

    assert result.returncode != 0
    assert "must remain unexecuted with no saved outputs" in result.stdout


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("invalid-json.ipynb", "{", "invalid JSON"),
        (
            "invalid-schema.ipynb",
            json.dumps(
                {
                    "nbformat": 4,
                    "nbformat_minor": 5,
                    "metadata": {},
                    "cells": [{"cell_type": "code"}],
                }
            ),
            "invalid notebook schema",
        ),
    ],
)
def test_rejects_invalid_notebook_documents(
    validation_project: Path,
    tmp_path: Path,
    name: str,
    content: str,
    message: str,
) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")

    result = validate(validation_project, path)

    assert result.returncode != 0
    assert message in result.stdout


def test_rejects_missing_execution_count(
    validation_project: Path, tmp_path: Path
) -> None:
    value = notebook()
    value.cells[1].execution_count = None
    path = tmp_path / "missing-count.ipynb"
    write_notebook(path, value)

    result = validate(validation_project, path)

    assert result.returncode != 0
    assert "code cell 1 has no execution count" in result.stdout


def test_rejects_stored_error(validation_project: Path, tmp_path: Path) -> None:
    value = notebook()
    value.cells[1].outputs = [
        new_output(
            "error",
            ename="RuntimeError",
            evalue="release example failed",
            traceback=[],
        )
    ]
    path = tmp_path / "error.ipynb"
    write_notebook(path, value)

    result = validate(validation_project, path)

    assert result.returncode != 0
    assert "code cell 1 stores RuntimeError: release example failed" in result.stdout


def test_rejects_missing_version_cell(validation_project: Path, tmp_path: Path) -> None:
    value = notebook(version_cells=0)
    path = tmp_path / "missing-version.ipynb"
    write_notebook(path, value)

    result = validate(validation_project, path)

    assert result.returncode != 0
    assert "missing torch_batteries.__version__ cell" in result.stdout


def test_rejects_multiple_version_cells(
    validation_project: Path, tmp_path: Path
) -> None:
    path = tmp_path / "multiple-version.ipynb"
    write_notebook(path, notebook(version_cells=2))

    result = validate(validation_project, path)

    assert result.returncode != 0
    assert "multiple torch_batteries.__version__ cells" in result.stdout


def test_rejects_stale_version_output(validation_project: Path, tmp_path: Path) -> None:
    value = notebook()
    value.cells[0].outputs[0].text = "torch-batteries version: 0.7.0\n"
    path = tmp_path / "stale-version.ipynb"
    write_notebook(path, value)

    result = validate(validation_project, path)

    assert result.returncode != 0
    assert f"does not report version {VERSION}" in result.stdout


def test_rejects_internal_import(validation_project: Path, tmp_path: Path) -> None:
    value = notebook()
    value.cells[1].source = "from torch_batteries.trainer.core import Battery"
    path = tmp_path / "internal-import.ipynb"
    write_notebook(path, value)

    result = validate(validation_project, path)

    assert result.returncode != 0
    assert "implementation-level" in result.stdout


def test_rejects_mismatched_project_versions(
    validation_project: Path, tmp_path: Path
) -> None:
    init_file = validation_project / "src" / "torch_batteries" / "__init__.py"
    init_file.write_text('__version__ = "0.7.0"\n', encoding="utf-8")
    path = tmp_path / "valid.ipynb"
    write_notebook(path, notebook())

    result = validate(validation_project, path)

    assert result.returncode != 0
    assert "project version mismatch" in result.stderr
