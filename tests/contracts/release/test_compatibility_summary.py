"""Check compatibility summaries report success, failure, and missing evidence."""

import importlib
import json
from pathlib import Path

import pytest


def test_success_summary_includes_environment_counts_and_all_stages(
    tmp_path: Path,
) -> None:
    summary = importlib.import_module("scripts.ci.compatibility_summary")
    (tmp_path / "environment.json").write_text(
        json.dumps(
            {
                "python": "3.13.9",
                "machine": "ARM64",
                "packages": {"torch": "2.15.0+cpu"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runtime.xml").write_text(
        '<testsuites><testsuite><testcase name="passed" time="1.2"/>'
        '<testcase name="optional" time="0"><skipped/></testcase>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    steps = {identifier: {"outcome": "success"} for identifier, _ in summary.STAGES}
    rendered = summary.render_summary(
        tmp_path,
        {"runner": "windows-11-arm", "architecture": "arm64", "python": "3.13"},
        steps,
    )
    assert "✅ **All compatibility checks passed" in rendered
    assert "windows-11-arm" in rendered
    assert "2.15.0+cpu" in rendered
    assert "| 1 | 0 | 0 | 1 | 1.20s |" in rendered
    assert "Installed-wheel smoke" in rendered


def test_failure_summary_includes_failed_tests_and_skipped_steps(
    tmp_path: Path,
) -> None:
    summary = importlib.import_module("scripts.ci.compatibility_summary")
    (tmp_path / "runtime.xml").write_text(
        '<testsuite><testcase classname="test_checkpoint" name="restore" time="0.5">'
        '<failure message="missing checkpoint">traceback</failure>'
        "</testcase></testsuite>",
        encoding="utf-8",
    )
    rendered = summary.render_summary(tmp_path, {}, {"runtime": {"outcome": "failure"}})
    assert "❌ **Compatibility checks failed" in rendered
    assert "test_checkpoint.restore" in rendered
    assert "missing checkpoint" in rendered
    assert "⏭️ Skipped" in rendered
    assert "| 0 | 1 | 0 | 0 | 0.50s |" in rendered


@pytest.mark.parametrize("report", [None, "invalid XML"])
def test_setup_failure_summary_does_not_claim_tests_passed(
    tmp_path: Path, report: str | None
) -> None:
    summary = importlib.import_module("scripts.ci.compatibility_summary")
    if report is not None:
        (tmp_path / "runtime.xml").write_text(report, encoding="utf-8")
    rendered = summary.render_summary(tmp_path, {}, {"python": {"outcome": "failure"}})
    assert "❌ **Compatibility checks failed" in rendered
    assert "JUnit report unavailable" in rendered
    assert "All compatibility checks passed" not in rendered


def test_summary_is_saved_as_artifact_and_appended_to_github_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = importlib.import_module("scripts.ci.compatibility_summary")
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "github-summary.md"
    destination.write_text("Earlier summary\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(destination))
    monkeypatch.setenv("CI_STEP_RESULTS", '{"runtime":{"outcome":"failure"}}')
    monkeypatch.setenv("CI_RUNNER", "ubuntu-latest")
    summary.main()
    artifact = tmp_path / "compatibility-results" / "summary.md"
    assert artifact.is_file()
    assert destination.read_text(
        encoding="utf-8"
    ) == "Earlier summary\n" + artifact.read_text(encoding="utf-8")
