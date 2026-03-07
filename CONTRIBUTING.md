# Contributing to torch-batteries

Thank you for your interest in contributing to torch-batteries! This guide will help you get started with development.

## Development

### Prerequisites
- Python 3.12+
- Pip 25.1+

### Setup Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/michalszc/torch-batteries.git
   cd torch-batteries
   ```

2. **Install development dependencies:**
   ```bash
   make install-dev
   ```
   This will:
   - Install the package in development mode
   - Install all development dependencies (pytest, ruff, mypy, pre-commit)
   - Set up pre-commit hooks

### Development Workflow

#### Available Make Commands
```bash
make help                # Show all available commands
make install             # Install package
make install-dev         # Install package with development dependencies
make test                # Run tests
make test-verbose        # Run tests with verbose output
make lint                # Run linting (ruff check)
make lint-fix            # Run linting with auto-fix
make format              # Format code (ruff format)
make format-check        # Check code formatting without making changes
make type-check          # Run type checking (mypy)
make build               # Build package for distribution
make publish-test        # Publish to TestPyPI
make publish             # Publish to PyPI
make check-build         # Check if build is ready for publishing
make validate-version    # Check if versions in pyproject.toml and __init__.py match
make clean               # Clean artifacts
```

#### Code Quality Tools

- **Linting & Formatting:** [Ruff](https://github.com/astral-sh/ruff) - Fast Python linter and formatter
- **Type Checking:** [MyPy](https://github.com/python/mypy) - Static type checker
- **Testing:** [pytest](https://pytest.org/) with coverage reporting
- **Pre-commit:** Automatic code quality checks before commits

#### Pre-commit Hooks
Pre-commit hooks are automatically installed with `make install-dev` and will run:
- Trailing whitespace removal
- End-of-file fixing
- YAML validation
- Ruff linting and formatting

To manually run pre-commit on all files:
```bash
pre-commit run --all-files
```

### Project Structure
```
torch-batteries/
├── src/torch_batteries/    # Main package source
│   ├── callbacks/          # Training callbacks
│   ├── events/             # Event system
│   ├── tracking/           # Experiment tracking
│   ├── trainer/            # Training system
│   └── utils/              # Utility functions
│       ├── progress/       # Progress tracking
│       └── ...             # Other utilities (batch, device, logging)
├── tests/                  # Test files (mirrors src/ structure)
│   ├── callbacks/          # Callbacks tests
│   ├── events/             # Event system tests
│   ├── tracking/           # Tracking tests
│   ├── trainer/            # Trainer tests
│   ├── utils/              # Utility tests
│   │   ├── progress/       # Progress tracking tests
│   │   └── ...             # Other utility tests
│   └── test_package.py     # Package-level tests
├── docs/                   # Generated documentation
├── notebooks/              # Jupyter notebooks for examples
├── assets/                 # Project assets
├── scripts/                # Build and CI scripts
│   ├── ci/                 # CI-specific scripts
│   └── validate_version.sh # Version validation script
├── .github/workflows/      # CI/CD pipelines
│   ├── ci.yml              # Continuous integration
│   ├── cd.yml              # Continuous deployment
│   └── docs.yml            # Documentation generation
├── pyproject.toml          # Project configuration
├── Makefile                # Development commands
├── LICENSE                 # MIT license
├── README.md               # User documentation
├── CONTRIBUTING.md         # This file
├── .gitignore              # Git ignore patterns
└── .pre-commit-config.yaml # Pre-commit hooks configuration
```

## Contributing Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and ensure code quality:**
   ```bash
   make lint
   make format-check
   make type-check
   make test
   make validate-version
   ```

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```
   (Pre-commit hooks will run automatically)

4. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

## CI/CD
The project uses GitHub Actions for continuous integration:
- **Quality Checks:** Tests, Linting, formatting, type checking, and version validation run in parallel
- **Publishing:** Automatic publishing to PyPI when merged to master

All checks must pass before merging pull requests.

## Release Process

1. **Update version** in both `pyproject.toml` and `src/torch_batteries/__init__.py`
2. **Validate version consistency:**
   ```bash
   make validate-version
   ```
3. **Run quality checks:**
    ```bash
    make lint
    make format-check
    make type-check
    make test
    ```
4. **Create a pull request** with your changes
5. **Merge to master** - this will automatically publish to PyPI

## Code Style

- Follow PEP 8 guidelines (enforced by Ruff)
- Use type hints for all functions and methods
- Write comprehensive tests for new features
- Keep test coverage above 90%
- Use descriptive variable and function names
- Add docstrings for public APIs

## Questions or Issues?

Feel free to open an issue on GitHub if you have any questions or run into problems during development.
