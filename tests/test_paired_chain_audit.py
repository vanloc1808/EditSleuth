from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_paired_chain_audit.py"
SPEC = importlib.util.spec_from_file_location("prepare_paired_chain_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extract_steps_accepts_plain_six_step_chain():
    text = "\n".join(f"{index}. step {index}" for index in range(1, 7))
    steps, valid = MODULE.extract_steps(text)
    assert valid
    assert steps[5] == "6. step 6"


def test_extract_steps_marks_missing_step_as_invalid():
    text = "\n".join(f"{index}. step {index}" for index in (1, 2, 4, 5, 6))
    steps, valid = MODULE.extract_steps(text)
    assert not valid
    assert steps[2] == ""


def test_consensus_tie_is_unclear():
    assert MODULE._consensus(pd.Series(["yes", "no"])) == "unclear"
