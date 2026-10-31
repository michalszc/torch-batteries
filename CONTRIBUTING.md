# Contributing to torch-batteries

Thank you for your interest in contributing to torch-batteries! This guide will help you get started with development.

## Development

### Prerequisites
- Python 3.12+
- Pip 25.1+
- [jq](https://jqlang.org/) (only for `make notebooks-check`; e.g. `sudo apt install jq` or `brew install jq`)

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

   **Alternative for [uv](https://docs.astral.sh/uv/) users**
   ```bash
   uv sync --all-extras
   uv run pre-commit install
   ```
   This creates a `.venv` with the same dependencies. Prefix commands with
   `uv run` (e.g. `uv run make test`), or activate the venv with
   `source .venv/bin/activate`. `uv.lock` is local only and not committed.

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
make validate-version    # Check that project versions match and have release notes
make validate-version-unpublished  # Also check that the version is newer than the latest on PyPI
make release-metadata    # Record the current commit and version without publishing
make notebooks-check     # Validate saved notebook outputs without executing them (requires jq)
make clean               # Clean artifacts
```

#### Code Quality Tools

- **Linting & Formatting:** [Ruff](https://github.com/astral-sh/ruff) - Fast Python linter and formatter
- **Type Checking:** [MyPy](https://github.com/python/mypy) - Static type checker
- **Testing:** [pytest](https://pytest.org/) with coverage reporting
- **Pre-commit:** Automatic code quality checks before commits

#### CPU workflow timing checks

`make test` also runs the performance tests in `tests/performance/`. They time
training, standalone validation, testing, prediction, and combined fit, with both
direct loaders and named DataPack datasets. Each case uses a tiny CPU-only model,
32 batches, one warmup, and three measured runs. The pytest summary reports median
total time and time per batch spent retrieving data, in the charged model step, and
in the remaining workflow. Training's remaining time includes PyTorch backward and
optimizer operations. Timings are informational and have no maximum-duration
failure threshold. The tests still check completed batch counts and valid timing
measurements. Compare the displayed timings from runs on the
same hardware to assess smaller changes. CUDA and MPS are never selected by these
tests. The per-batch columns use microseconds (µs/batch); total time uses
milliseconds (ms).

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
│   │   └── run_check.sh    # CI check dispatcher
│   ├── validate_notebooks.sh # Static notebook checks
│   └── validate_version.sh # Version and release checks
├── .github/workflows/      # CI/CD pipelines
│   ├── ci.yml              # Continuous integration
│   ├── cd.yml              # Package build for every master commit
│   ├── release.yml         # Manual PyPI and GitHub release publication
│   └── docs.yml            # Documentation generation
├── pyproject.toml          # Project configuration
├── Makefile                # Development commands
├── LICENSE                 # Apache License 2.0
├── NOTICE                  # Project copyright notice
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
- **Builds:** Successful CI on master triggers CD, which builds and checks the package without publishing it
- **Publishing:** A maintainer runs the Release workflow manually to publish master's CD build to PyPI, followed by a GitHub release

All checks must pass before merging pull requests.

## Release Process

This section is for maintainers. Every master commit goes through **CI → CD** automatically, which builds the package without publishing it. Releasing is a separate manual decision made by running the **Release** workflow.

1. **Prepare a new package version** in both `pyproject.toml` and `src/torch_batteries/__init__.py`, and add exactly one matching version section with nonempty notes in `documentation/release-notes.md`.
2. **Validate the proposed version** with `make validate-version-unpublished`. CI runs `make validate-version`, which checks consistency and release notes but not PyPI, so pull requests do not need a version bump.
3. **Run the existing quality checks** using the development environment:
   ```bash
   source .venv/bin/activate
   make lint
   make format-check
   make type-check
   make test
   ```
   These targets cover `scripts/release.py` and the release contract tests; no separate release-check target is needed.
4. **Create a pull request and merge to master.** Successful CI from a push to this repository's master branch triggers CD. CD checks out that exact CI commit, builds and checks the package, and uploads two artifacts named after the commit: `dist-<sha>` with the built files and `release-metadata-<sha>` with its commit SHA, package name, and version. CD never publishes, so packaging problems surface right after merge.
5. **Run the Release workflow** from the Actions tab (**Release → Run workflow**, branch `master`), or with `gh workflow run release.yml --ref master`. By default it releases master HEAD. The optional `commit` input selects an earlier master commit instead, such as the version-bump commit when later commits were skipped by CI's path filters and therefore have no CD build; `make release-resolve-commit` rejects commits that are not on master.

The Release workflow's `publish` job runs in the `pypi` deployment environment. `make release-find-build` selects the newest CD run that uploaded unexpired artifacts for the commit; without one, for example because CI failed, CD has not finished, or CI did not run for the commit, the release stops. The job downloads that exact build, `make release-verify-build` checks that its metadata matches the checkout, `make validate-version-unpublished` rejects a version that is not newer than the latest on PyPI, so neither a repeated release nor an older commit can be published, and the downloaded files are checked and uploaded to PyPI without rebuilding. Only then does the `github-release` job run `make release`, which creates the `v<version>` tag on that commit and the GitHub release.

The release title and tag are `v<version>`. Its body starts with `## What's Changed`, followed by the matching curated release-notes list and a **View on PyPI** link to that version. GitHub automatically provides source ZIP and tar.gz archives for the tag; wheel and package source-distribution files remain on PyPI.

Running a workflow manually requires write access to the repository. Adding required reviewers to the `pypi` environment in the repository settings restricts publication to maintainers, and restricting the environment to the `master` branch duplicates the workflow's own branch check. The `publish` job uses the `PYPI_API_TOKEN` secret and the automatic `GITHUB_TOKEN` with `actions: read` to find and download CD artifacts; the `github-release` job additionally has `contents: write` to create the tag and release.

### Local release tooling

To inspect the current checkout's deployment identity without publishing:

```bash
source .venv/bin/activate
make release-metadata
# Optional output path:
make release-metadata RELEASE_METADATA=/tmp/release-metadata.json
make release-verify-build RELEASE_METADATA=/tmp/release-metadata.json
```

CD supplies `DEPLOYED_SHA` so recording metadata also verifies the expected CI commit. Local use defaults to the current checkout. The default generated `release-metadata.json` is ignored by Git.

`make release-resolve-commit` accepts `RELEASE_COMMIT` (default `HEAD`) and writes the full SHA of that commit on the current branch to `GITHUB_OUTPUT`. `make release-find-build` needs GitHub CLI authentication, `RELEASE_SHA`, and a repository specified through `GITHUB_REPOSITORY` or `RELEASE_REPOSITORY`; it writes the selected run ID to `GITHUB_OUTPUT`. `make release` publishes through GitHub CLI and requires authentication, matching metadata, and a repository specified the same way. `RELEASE_METADATA` selects the metadata file and `RELEASE_PROJECT` selects the checkout directory (also honoured by `make validate-version-unpublished`). The Release workflow runs master's tooling from the repository root and checks the release commit out into `release-src`, so older commits can be released with current release logic. Neither publishing nor package builds are needed to run the four quality checks above.

### Recovery

If the release stops before uploading to PyPI, nothing was published: fix the cause on master and run the Release workflow again. A failed CD run can be rerun, but GitHub only allows rerunning a workflow run within 30 days of its start, while CD artifacts expire after the repository's artifact retention period (90 days by default). An expired build therefore cannot be recreated: push a new commit to master, for example the next version bump, and release that commit instead. If PyPI publication succeeded but the `github-release` job failed, correct the cause and use **Re-run failed jobs**, which reruns only `github-release` with the same CD artifacts while they remain available. Do not start a new Release run for that version, since its PyPI validation fails once the version is published.

Release runs are serialized without cancelling active publication. A retry skips an already published release if its tag resolves to the same commit. Conflicting tags and existing draft releases fail for maintainer review; automation never moves tags or overwrites releases. Check the job logs and summary for the outcome and release URL.

## Code Style

- Follow PEP 8 guidelines (enforced by Ruff)
- Use type hints for all functions and methods
- Write comprehensive tests for new features
- Keep test coverage above 90%
- Use descriptive variable and function names
- Add docstrings for public APIs

## Questions or Issues?

Feel free to open an issue on GitHub if you have any questions or run into problems during development.
