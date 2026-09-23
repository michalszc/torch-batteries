"""CPU workflow timing checks included in the normal unit test suite."""

import pytest

from tests.performance.benchmark import MODES, PHASES, Measurement, measure

MEASUREMENTS: dict[str, Measurement] = {}
MAX_SCENARIO_MS = 30.0


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("mode", MODES)
def test_each_workflow_records_all_batches_and_isolates_costs(
    phase: str, mode: str
) -> None:
    result = measure(phase, mode, batches=32, batch_size=1, repeats=3, warmups=1)
    MEASUREMENTS[f"{mode}/{phase}"] = result
    assert result.batches == 32 * (2 if mode == "named" else 1) * (
        2 if phase == "fit" else 1
    )
    assert result.total_ms > 0
    assert result.loader_us_per_batch >= 0
    assert result.step_us_per_batch >= 0
    assert result.overhead_us_per_batch >= 0
    assert result.total_ms < MAX_SCENARIO_MS, (
        f"{mode}/{phase} took {result.total_ms:.1f} ms; "
        f"the CPU limit is {MAX_SCENARIO_MS:.0f} ms"
    )
