# torch-batteries

A lightweight Python package that supplies batteries-included abstractions for:
- Data loading pipelines
- Model training loops
- Evaluation workflows
- Metrics computation
- Seamless Weights & Biases tracking

Designed to reduce boilerplate and standardize experiment code.

## Installation

### For Users
```bash
pip install torch-batteries
```

### For Development
```bash
git clone https://github.com/michalszc/torch-batteries.git
cd torch-batteries
make install-dev
```

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
make install             # Install package only
make install-dev         # Install with development dependencies
make test                # Run tests with coverage
make test-verbose        # Run tests with verbose output
make lint                # Run linting (ruff check)
make lint-fix            # Run linting with auto-fix
make format              # Format code (ruff format)
make format-check        # Check code formatting
make type-check          # Run type checking (mypy)
make clean               # Clean build artifacts and caches
make build               # Build distribution packages
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
- MyPy type checking

To manually run pre-commit on all files:
```bash
pre-commit run --all-files
```

### Project Structure
```
torch-batteries/
├── src/torch_batteries/    # Main package source
│   └── __init__.py         # Package initialization
├── tests/                  # Test files
│   └── test_package.py     # Package tests
├── notebooks/              # Jupyter notebooks for examples
├── .github/workflows/      # CI/CD pipelines
├── pyproject.toml          # Project configuration
├── Makefile                # Development commands
└── README.md               # This file
```

### Contributing

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and ensure code quality:**
   ```bash
   make lint
   make test
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

### CI/CD
The project uses GitHub Actions for continuous integration:
- **Linting:** Ruff checks code quality
- **Type Checking:** MyPy validates type hints
- **Testing:** pytest runs all tests with coverage
- **Multi-Python:** Tested on Python 3.12+

All checks must pass before merging pull requests.
