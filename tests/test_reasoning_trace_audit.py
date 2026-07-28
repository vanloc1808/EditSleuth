from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_reasoning_traces.py"
SPEC = importlib.util.spec_from_file_location("audit_reasoning_traces", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_split_six_steps_ignores_instruction_internal_numbering():
    chain = (
        "1. The instruction says:\n"
        "1.) Buy groceries\n"
        "2.) Call Mom.\n"
        "2. The mask covers 20%.\n"
        "3. Structural change is minor.\n"
        "4. Category is text_edit.\n"
        "5. Text edits often have rendering artifacts.\n"
        "6. Overall difficulty is medium."
    )
    steps = MODULE.split_six_steps(chain)
    assert len(steps) == 6
    assert "2.) Call Mom" in steps[0]
    assert steps[1] == "2. The mask covers 20%."


def test_allocation_is_balanced_and_totals_200():
    allocation = MODULE._allocation(200)
    assert allocation == {"easy": 67, "medium": 67, "hard": 66}
    assert sum(allocation.values()) == 200
