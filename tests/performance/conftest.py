"""Show CPU performance measurements in the ordinary pytest summary."""

from typing import TYPE_CHECKING

from tests.performance.test_workflows import MEASUREMENTS

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter


def pytest_terminal_summary(terminalreporter: "TerminalReporter") -> None:
    """Report medians even when pytest captures passing test output."""
    if not MEASUREMENTS:
        return
    terminalreporter.write_sep("=", "CPU workflow timings (median, 32 batches)")
    terminalreporter.write_line(
        "scenario          total ms  loader µs/batch  step µs/batch  workflow µs/batch"
    )
    for name, values in sorted(MEASUREMENTS.items()):
        terminalreporter.write_line(
            f"{name:18} {values.total_ms:9.2f} "
            f"{values.loader_us_per_batch:13.1f} "
            f"{values.step_us_per_batch:11.1f} "
            f"{values.overhead_us_per_batch:15.1f}"
        )
