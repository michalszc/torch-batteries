#!/usr/bin/env bash
# Validate version consistency across project files, release notes, and PyPI

set -e

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

# Check PyPI version
PACKAGE_NAME="torch-batteries"
PYPI_VERSION=$(curl -s "https://pypi.org/pypi/${PACKAGE_NAME}/json" | python -c "import sys, json; data = json.load(sys.stdin); print(data['info']['version'])" 2>/dev/null || echo "")

if [ -z "$PYPI_VERSION" ]; then
    echo "ℹ️  Package '${PACKAGE_NAME}' not found on PyPI (first release?)"
else
    echo "📦 PyPI:           $PYPI_VERSION"

    if [ "$PYPROJECT_VERSION" = "$PYPI_VERSION" ]; then
        echo ""
        echo "❌ Version matches PyPI! Please bump the version before publishing."
        exit 1
    else
        echo ""
        echo "✅ Version $PYPROJECT_VERSION is different from PyPI ($PYPI_VERSION)"
    fi
fi

echo ""
echo "✨ All checks passed! Ready to publish version $PYPROJECT_VERSION"
