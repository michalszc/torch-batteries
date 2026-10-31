#!/usr/bin/env bash
# Validate version consistency across project files and release notes.
# With --unpublished, also require that the version is newer than the latest on PyPI.

set -e

# RELEASE_PROJECT lets master's tooling validate a separately checked-out release commit.
cd "${RELEASE_PROJECT:-.}"

CHECK_PYPI=false
if [ "${1:-}" = "--unpublished" ]; then
    CHECK_PYPI=true
fi

echo "🔍 Validating version consistency..."
echo ""

# Get version from pyproject.toml
PYPROJECT_VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
echo "📄 pyproject.toml: $PYPROJECT_VERSION"

# Get version from __init__.py
INIT_VERSION=$(python -c "import sys; sys.path.insert(0, 'src'); import torch_batteries; print(torch_batteries.__version__)")
echo "📄 __init__.py:    $INIT_VERSION"

# Check if versions match
if [ "$PYPROJECT_VERSION" != "$INIT_VERSION" ]; then
    echo ""
    echo "❌ Version mismatch between pyproject.toml and __init__.py!"
    exit 1
fi

echo ""
echo "✅ Versions match in project files"

# Check release notes
RELEASE_NOTES="documentation/release-notes.md"
if [ ! -f "$RELEASE_NOTES" ]; then
    echo ""
    echo "❌ Release notes file not found: $RELEASE_NOTES"
    exit 1
fi

RELEASE_NOTE_COUNT=$(python -c 'import re, sys; version, path = sys.argv[1:]; lines = open(path, encoding="utf-8").read().splitlines(); print(sum(re.match(r"^##\s+" + re.escape(version) + r"(?:\s|$)", line) is not None for line in lines))' "$PYPROJECT_VERSION" "$RELEASE_NOTES")

if [ "$RELEASE_NOTE_COUNT" -eq 0 ]; then
    echo ""
    echo "❌ Version $PYPROJECT_VERSION is missing from $RELEASE_NOTES"
    exit 1
fi

if [ "$RELEASE_NOTE_COUNT" -ne 1 ]; then
    echo ""
    echo "❌ Version $PYPROJECT_VERSION appears multiple times in $RELEASE_NOTES"
    exit 1
fi

echo "📋 Release notes:   $PYPROJECT_VERSION"
echo "✅ Current version is included exactly once in release notes"

if [ "$CHECK_PYPI" = true ]; then
    # Check PyPI version
    PACKAGE_NAME="torch-batteries"
    PYPI_JSON=$(mktemp)
    trap 'rm -f "$PYPI_JSON"' EXIT
    # Only a 404 means "never published"; any other failure must stop the release.
    if ! PYPI_STATUS=$(curl -sS --retry 3 -o "$PYPI_JSON" -w '%{http_code}' "https://pypi.org/pypi/${PACKAGE_NAME}/json"); then
        echo ""
        echo "❌ Could not reach PyPI to check the latest version"
        exit 1
    fi

    PYPI_VERSION=""
    if [ "$PYPI_STATUS" != "404" ]; then
        if [ "$PYPI_STATUS" != "200" ]; then
            echo ""
            echo "❌ PyPI returned HTTP $PYPI_STATUS while checking the latest version"
            exit 1
        fi
        if ! PYPI_VERSION=$(python -c "import sys, json; print(json.load(open(sys.argv[1]))['info']['version'])" "$PYPI_JSON"); then
            echo ""
            echo "❌ Could not read the latest version from the PyPI response"
            exit 1
        fi
    fi

    if [ -z "$PYPI_VERSION" ]; then
        echo "ℹ️  Package '${PACKAGE_NAME}' not found on PyPI (first release?)"
    else
        echo "📦 PyPI:           $PYPI_VERSION"

        if ! python -c 'import sys; from packaging.version import Version; sys.exit(Version(sys.argv[1]) <= Version(sys.argv[2]))' "$PYPROJECT_VERSION" "$PYPI_VERSION"; then
            echo ""
            echo "❌ Version $PYPROJECT_VERSION is not newer than PyPI ($PYPI_VERSION)! Please bump the version before publishing."
            exit 1
        else
            echo ""
            echo "✅ Version $PYPROJECT_VERSION is newer than PyPI ($PYPI_VERSION)"
        fi
    fi
fi

echo ""
echo "✨ All checks passed for version $PYPROJECT_VERSION"
