"""Write portable GitHub job summaries from compatibility stages and JUnit data."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from typing import cast

STAGES = (
    ("python", "Native Python setup"),
    ("dependencies", "Test/build dependencies"),
    ("pytorch", "CPU PyTorch installation"),
    ("build", "Wheel build"),
    ("install", "Wheel installation"),
    ("pip_check", "Dependency consistency"),
    ("runtime", "Runtime tests"),
    ("smoke", "Installed-wheel smoke"),
    ("environment", "Environment report"),
)
BADGES = {
    "success": "✅ Passed",
    "failure": "❌ Failed",
    "skipped": "⏭️ Skipped",
    "cancelled": "⚠️ Cancelled",
}


def _cell(value: object) -> str:
    """Keep report values on one Markdown table row."""
    return escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def _test_summary(path: Path) -> list[str]:
    """Summarize individual test cases without double-counting suite totals."""
    try:
        cases = list(ET.parse(path).getroot().iter("testcase"))
        failures = [case for case in cases if case.find("failure") is not None]
        errors = [case for case in cases if case.find("error") is not None]
        skipped = sum(case.find("skipped") is not None for case in cases)
        seconds = sum(float(case.get("time", "0")) for case in cases)
    except (OSError, ET.ParseError, ValueError):
        return ["JUnit report unavailable; inspect the stage results and job logs.", ""]
    passed = len(cases) - len(failures) - len(errors) - skipped
    lines = [
        "### 🧪 Runtime test results",
        "",
        "| Passed | Failed | Errors | Skipped | Duration |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {passed} | {len(failures)} | {len(errors)} | {skipped} | {seconds:.2f}s |",
        "",
    ]
    if failures or errors:
        lines.extend(["### ❌ Failing tests (first 10)", ""])
        for case in (failures + errors)[:10]:
            problem = case.find("failure")
            if problem is None:
                problem = case.find("error")
            message = (
                problem.get("message", "See job logs")
                if problem is not None
                else "See job logs"
            )
            name = f"{case.get('classname', '')}.{case.get('name', 'unknown')}".lstrip(
                "."
            )
            lines.append(f"- **{_cell(name)}**: {_cell(message[:500])}")
        lines.extend(
            [
                "",
                "💡 **Fix suggestion:** inspect the failing tests and full job logs.",
                "",
            ]
        )
    return lines


def render_summary(
    directory: Path, identity: dict[str, str], steps: dict[str, dict[str, object]]
) -> str:
    """Render environment, stage status, and test evidence for one matrix cell."""
    outcomes = {
        identifier: str(steps.get(identifier, {}).get("outcome", "skipped"))
        for identifier, _ in STAGES
    }
    if "failure" in outcomes.values():
        status = "❌ **Compatibility checks failed.**"
    elif all(outcome == "success" for outcome in outcomes.values()):
        status = "✅ **All compatibility checks passed.**"
    else:
        status = "⚠️ **Compatibility checks incomplete.**"
    try:
        environment = json.loads(
            (directory / "environment.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        environment = {}
    torch_version = environment.get("packages", {}).get("torch", "Not available")
    lines = [
        "## 🧪 Native Compatibility",
        "",
        status,
        "",
        "| Environment | Value |",
        "| --- | --- |",
        f"| Runner | {_cell(identity.get('runner', 'Not available'))} |",
        f"| Architecture | {_cell(identity.get('architecture', 'Not available'))} |",
        f"| Python target | {_cell(identity.get('python', 'Not available'))} |",
        f"| Actual Python | {_cell(environment.get('python', 'Not available'))} |",
        f"| PyTorch | {_cell(torch_version)} |",
        "",
        "### Check results",
        "",
        "| Stage | Result |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| {label} | {BADGES.get(outcomes[identifier], '❔ Unknown')} |"
        for identifier, label in STAGES
    )
    lines.extend(["", *_test_summary(directory / "runtime.xml")])
    lines.extend(
        [
            (
                "CPU compatibility covers the tested dependency versions. "
                "CUDA, MPS, and native Inductor compilation are outside this matrix."
            ),
            "",
            (
                "Environment details, JUnit results, and this summary are saved "
                "in the compatibility artifact."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Append a summary to GitHub and retain the same Markdown as an artifact."""
    directory = Path("compatibility-results")
    directory.mkdir(parents=True, exist_ok=True)
    steps = cast(
        "dict[str, dict[str, object]]", json.loads(os.getenv("CI_STEP_RESULTS", "{}"))
    )
    identity = {
        "runner": os.getenv("CI_RUNNER", "Not available"),
        "architecture": os.getenv("CI_ARCHITECTURE", "Not available"),
        "python": os.getenv("CI_PYTHON", "Not available"),
    }
    summary = render_summary(directory, identity, steps)
    (directory / "summary.md").write_text(summary, encoding="utf-8")
    if destination := os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
