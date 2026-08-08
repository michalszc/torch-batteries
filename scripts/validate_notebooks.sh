#!/usr/bin/env bash

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEFAULT_PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$DEFAULT_PROJECT_ROOT

if [ "${1:-}" = "--project-root" ]; then
    if [ "$#" -lt 2 ]; then
        printf '%s\n' "usage: $0 [--project-root PATH] [NOTEBOOK_OR_DIRECTORY ...]" >&2
        exit 2
    fi
    PROJECT_ROOT=$(CDPATH= cd -- "$2" && pwd) || exit 2
    shift 2
fi

if ! command -v jq >/dev/null 2>&1; then
    printf '%s\n' "notebook validation requires jq" >&2
    exit 2
fi

if [ -x "$DEFAULT_PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON=$DEFAULT_PROJECT_ROOT/.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=$(command -v python3)
elif command -v python >/dev/null 2>&1; then
    PYTHON=$(command -v python)
else
    printf '%s\n' "notebook validation requires Python" >&2
    exit 2
fi

PYPROJECT="$PROJECT_ROOT/pyproject.toml"
INIT_FILE="$PROJECT_ROOT/src/torch_batteries/__init__.py"

if [ ! -f "$PYPROJECT" ] || [ ! -f "$INIT_FILE" ]; then
    printf '%s\n' "project root must contain pyproject.toml and src/torch_batteries/__init__.py" >&2
    exit 2
fi

pyproject_version=$(
    "$PYTHON" -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["version"])' "$PYPROJECT"
) || exit 2
init_version=$(
    "$PYTHON" -c 'import ast, sys; tree=ast.parse(open(sys.argv[1], encoding="utf-8").read()); values=[ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]; print(values[0] if len(values) == 1 else "")' "$INIT_FILE"
) || exit 2

if [ -z "$init_version" ] || [ "$pyproject_version" != "$init_version" ]; then
    printf 'project version mismatch: pyproject.toml=%s, __init__.py=%s\n' \
        "$pyproject_version" "${init_version:-missing}" >&2
    exit 1
fi

runtime_version=$(
    PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON" -c 'import torch_batteries; print(torch_batteries.__version__)'
) || exit 2
if [ "$runtime_version" != "$pyproject_version" ]; then
    printf 'project version mismatch: declared=%s, imported=%s\n' \
        "$pyproject_version" "$runtime_version" >&2
    exit 1
fi

notebooks=()
add_notebooks() {
    candidate=$1
    if [ -d "$candidate" ]; then
        while IFS= read -r -d '' notebook; do
            notebooks+=("$notebook")
        done < <(find "$candidate" -maxdepth 1 -type f -name '*.ipynb' -print0)
    elif [ -f "$candidate" ] && [ "${candidate##*.}" = "ipynb" ]; then
        notebooks+=("$candidate")
    else
        printf 'notebook path does not exist or is not an .ipynb file: %s\n' "$candidate" >&2
        exit 2
    fi
}

if [ "$#" -eq 0 ]; then
    add_notebooks "$PROJECT_ROOT/notebooks"
else
    for candidate in "$@"; do
        add_notebooks "$candidate"
    done
fi

if [ "${#notebooks[@]}" -eq 0 ]; then
    printf '%s\n' "no notebooks found" >&2
    exit 1
fi

failed=0
for notebook in "${notebooks[@]}"; do
    name=$(basename -- "$notebook")
    reason=

    if ! jq empty "$notebook" >/dev/null 2>&1; then
        reason="invalid JSON"
    elif ! "$PYTHON" -c 'import nbformat, sys; nbformat.validate(nbformat.read(sys.argv[1], as_version=4))' "$notebook" >/dev/null 2>&1; then
        reason="invalid notebook schema"
    elif ! jq -e '.nbformat == 4 and (.cells | type == "array")' "$notebook" >/dev/null; then
        reason="notebook format must be version 4 with a cells array"
    fi

    if [ -z "$reason" ]; then
        invalid_installation=$(jq -r '
            .cells | to_entries[] |
            select(.value.cell_type == "code") |
            select(
                (.value.source | if type == "array" then join("") else . end) |
                test("^\\s*%pip\\s+install\\s+.*torch-batteries")
            ) |
            select(
                .value.execution_count != null or
                ((.value.outputs // []) | length) != 0
            ) |
            .key
        ' "$notebook" | head -n 1)
        if [ -n "$invalid_installation" ]; then
            reason="installation cell $invalid_installation must remain unexecuted with no saved outputs"
        fi
    fi

    if [ -z "$reason" ]; then
        missing_execution=$(jq -r '
            .cells | to_entries[] |
            select(.value.cell_type == "code") |
            select(.value.execution_count == null) |
            select(
                (.value.source | if type == "array" then join("") else . end) |
                test("^\\s*%pip\\s+install\\s+.*torch-batteries") | not
            ) |
            .key
        ' "$notebook" | head -n 1)
        if [ -n "$missing_execution" ]; then
            reason="code cell $missing_execution has no execution count"
        fi
    fi

    if [ -z "$reason" ]; then
        stored_error=$(jq -r '
            .cells | to_entries[] as $cell |
            $cell.value.outputs[]? |
            select(.output_type == "error") |
            "code cell \($cell.key) stores \(.ename // "error"): \(.evalue // "")"
        ' "$notebook" | head -n 1)
        if [ -n "$stored_error" ]; then
            reason=$stored_error
        fi
    fi

    if [ -z "$reason" ]; then
        version_cell_count=$(jq '[
            .cells[] |
            select(.cell_type == "code") |
            (.source | if type == "array" then join("") else . end) |
            select(contains("torch_batteries.__version__"))
        ] | length' "$notebook")
        if [ "$version_cell_count" -eq 0 ]; then
            reason="missing torch_batteries.__version__ cell"
        elif [ "$version_cell_count" -ne 1 ]; then
            reason="multiple torch_batteries.__version__ cells"
        fi
    fi

    if [ -z "$reason" ]; then
        version_output=$(jq -r '
            .cells[] |
            select(.cell_type == "code") |
            select((.source | if type == "array" then join("") else . end) | contains("torch_batteries.__version__")) |
            .outputs[]? |
            if .output_type == "stream" then .text
            elif .output_type == "display_data" or .output_type == "execute_result" then .data["text/plain"] // empty
            else empty end |
            if type == "array" then join("") else . end
        ' "$notebook")
        escaped_version=${pyproject_version//./\.}
        if ! printf '%s\n' "$version_output" | grep -Eq "(^|[^0-9.])${escaped_version}([^0-9.]|$)"; then
            reason="saved output does not report version $pyproject_version"
        fi
    fi

    if [ -z "$reason" ]; then
        internal_import=$(jq -r '
            .cells[] |
            select(.cell_type == "code") |
            (.source | if type == "array" then join("") else . end) |
            select(test("torch_batteries\\.(events\\.core|trainer\\.core|callbacks\\.[a-z_][a-z0-9_]*)"))
        ' "$notebook" | head -n 1)
        if [ -n "$internal_import" ]; then
            reason="uses an implementation-level torch_batteries import"
        fi
    fi

    if [ -n "$reason" ]; then
        printf '✗ %s: %s\n' "$name" "$reason"
        failed=1
    else
        printf '✓ %s: executed with torch-batteries %s\n' "$name" "$pyproject_version"
    fi
done

exit "$failed"
