.PHONY: help install install-dev test lint lint-fix format format-check type-check build clean validate-version
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e . --group dev
	pre-commit install

test: ## Run tests
	pytest tests/ --cov=src/torch_batteries --cov-report=term-missing --cov-report=html

test-verbose: ## Run tests with verbose output
	pytest tests/ -v --cov=src/torch_batteries --cov-report=term-missing

lint: ## Run linting (ruff check)
	ruff check src/ tests/

lint-fix: ## Run linting with auto-fix
	ruff check --fix src/ tests/

format: ## Format code (ruff format)
	ruff format src/ tests/

format-check: ## Check code formatting without making changes
	ruff format --diff src/ tests/

type-check: ## Run type checking (mypy)
	mypy src/torch_batteries/ tests/

build: ## Build package for distribution
	python -m build

publish-test: ## Publish to TestPyPI
	python -m twine upload --repository testpypi dist/*

publish: ## Publish to PyPI
	python -m twine upload dist/*

check-build: ## Check if build is ready for publishing
	python -m twine check dist/*

validate-version: ## Check if versions in pyproject.toml and __init__.py match
	@PYPROJECT_VERSION=$$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"); \
	INIT_VERSION=$$(python -c "import sys; sys.path.insert(0, 'src'); import torch_batteries; print(torch_batteries.__version__)"); \
	if [ "$$PYPROJECT_VERSION" = "$$INIT_VERSION" ]; then \
		echo "✅ Versions match: $$PYPROJECT_VERSION"; \
	else \
		echo "❌ Version mismatch:"; \
		echo "  pyproject.toml: $$PYPROJECT_VERSION"; \
		echo "  __init__.py:    $$INIT_VERSION"; \
		exit 1; \
	fi

clean: ## Clean artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
