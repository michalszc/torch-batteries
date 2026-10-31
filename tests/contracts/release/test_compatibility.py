"""Contracts for required native Python/platform compatibility coverage."""

import itertools
import tomllib

from tests.contracts.release.test_workflows import ROOT, workflow


def test_required_native_matrix_runs_after_quality_checks() -> None:
    job = workflow("ci")["jobs"]["compatibility-tests"]
    assert job["needs"] == "quality-checks"
    assert job.get("continue-on-error", "false") == "false"
    assert job["strategy"]["fail-fast"] == "false"
    matrix = job["strategy"]["matrix"]
    actual = {
        (platform["os"], platform["architecture"], version)
        for platform, version in itertools.product(
            matrix["platform"], matrix["python-version"]
        )
        if {"platform": platform, "python-version": version} not in matrix["exclude"]
    }
    expected = {
        (runner, architecture, version)
        for runner, architecture in [
            ("ubuntu-latest", "x64"),
            ("macos-latest", "arm64"),
            ("windows-latest", "x64"),
            ("windows-11-arm", "arm64"),
        ]
        for version in ("3.12", "3.13", "3.14")
        if (runner, version) != ("windows-11-arm", "3.14")
    }
    assert actual == expected
    assert len(actual) == 11
    assert job["runs-on"] == "${{ matrix.platform.os }}"
    setup = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    assert setup["with"]["architecture"] == "${{ matrix.platform.architecture }}"
    assert setup["with"]["python-version"] == "${{ matrix.python-version }}"
    assert setup["with"]["cache"] == "pip"


def test_matrix_checks_installed_wheel_and_runtime_workflows() -> None:
    steps = workflow("ci")["jobs"]["compatibility-tests"]["steps"]
    commands = {step.get("name"): step.get("run", "") for step in steps}
    assert "https://download.pytorch.org/whl/cpu" in commands["Install CPU PyTorch"]
    assert "--only-binary=:all:" in commands["Install CPU PyTorch"]
    assert "--wheel" in commands["Build wheel"]
    assert "scripts/ci/compatibility.py install-wheel" in commands["Install wheel"]
    assert "pip check" in commands["Check installed dependencies"]
    assert (
        "python -I scripts/ci/smoke_installed_package.py"
        in commands["Smoke test installed wheel"]
    )
    runtime = commands["Run runtime tests"]
    for directory in (
        "callbacks",
        "data",
        "events",
        "tracking",
        "trainer",
        "utils",
        "performance",
    ):
        assert f"tests/{directory}" in runtime
    assert '"not inductor"' in runtime
    assert "--junitxml=" in runtime
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["if"] == "always()"
    for identity in (
        "matrix.platform.os",
        "matrix.platform.architecture",
        "matrix.python-version",
    ):
        assert identity in upload["with"]["name"]
    assert (
        next(step for step in steps if step.get("name") == "Record environment")["if"]
        == "always()"
    )


def test_matrix_summary_runs_even_after_failures() -> None:
    steps = workflow("ci")["jobs"]["compatibility-tests"]["steps"]
    summary = next(
        step for step in steps if step.get("name") == "Write compatibility summary"
    )
    assert summary["if"] == "always()"
    assert "scripts/ci/compatibility_summary.py" in summary["run"]
    assert "toJSON(steps)" in summary["env"]["CI_STEP_RESULTS"]


def test_matrix_has_no_pip_upgrade_or_inline_python() -> None:
    steps = workflow("ci")["jobs"]["compatibility-tests"]["steps"]
    for step in steps:
        assert step.get("shell") != "python"
        assert "--upgrade pip" not in step.get("run", "")
        assert "--group" not in step.get("run", "")
    commands = {step.get("name"): step.get("run", "") for step in steps}
    assert (
        "scripts/ci/compatibility.py install-test-dependencies"
        in commands["Install test and build dependencies"]
    )
    assert (
        "scripts/ci/compatibility.py record-environment"
        in commands["Record environment"]
    )


def test_compatibility_helpers_trigger_ci_and_python_versions_are_advertised() -> None:
    config = workflow("ci")
    for event in ("push", "pull_request"):
        assert "scripts/ci/**" in config["on"][event]["paths"]
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    assert project["requires-python"] == ">=3.12"
    for version in ("3.12", "3.13", "3.14"):
        assert f"Programming Language :: Python :: {version}" in project["classifiers"]
