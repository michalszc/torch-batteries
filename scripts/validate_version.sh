#!/usr/bin/env bash
# Validate version consistency across project files and PyPI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Validating version consistency...${NC}\n"

# Get version from pyproject.toml
PYPROJECT_VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
echo -e "${BLUE}📄 pyproject.toml:${NC} $PYPROJECT_VERSION"

# Get version from __init__.py
INIT_VERSION=$(python -c "import sys; sys.path.insert(0, 'src'); import torch_batteries; print(torch_batteries.__version__)")
echo -e "${BLUE}📄 __init__.py:   ${NC} $INIT_VERSION"

# Check if versions match
if [ "$PYPROJECT_VERSION" != "$INIT_VERSION" ]; then
    echo -e "\n${RED}❌ Version mismatch between pyproject.toml and __init__.py!${NC}"
    exit 1
fi

echo -e "\n${GREEN}✅ Versions match in project files${NC}"

# Check PyPI version
PACKAGE_NAME="torch-batteries"
PYPI_VERSION=$(curl -s "https://pypi.org/pypi/${PACKAGE_NAME}/json" | python -c "import sys, json; data = json.load(sys.stdin); print(data['info']['version'])" 2>/dev/null || echo "")

if [ -z "$PYPI_VERSION" ]; then
    echo -e "${YELLOW}ℹ️  Package '${PACKAGE_NAME}' not found on PyPI (first release?)${NC}"
else
    echo -e "${BLUE}📦 PyPI:          ${NC} $PYPI_VERSION"

    if [ "$PYPROJECT_VERSION" = "$PYPI_VERSION" ]; then
        echo -e "\n${RED}❌ Version matches PyPI! Please bump the version before publishing.${NC}"
        exit 1
    else
        echo -e "\n${GREEN}✅ Version $PYPROJECT_VERSION is different from PyPI ($PYPI_VERSION)${NC}"
    fi
fi

echo -e "\n${GREEN}✨ All checks passed! Ready to publish version $PYPROJECT_VERSION${NC}"
