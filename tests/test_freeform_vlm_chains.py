from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_freeform_vlm_chains.py"
SPEC = importlib.util.spec_from_file_location("generate_freeform_vlm_chains", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_difficulty_sample_balances_200_rows():
    rows = []
    for bin_name in ("easy", "medium", "hard"):
        for index in range(100):
            rows.append({
                "triplet_id": f"{bin_name}-{index}",
                "reasoning_difficulty_bin": bin_name,
            })
    sampled = MODULE.difficulty_sample(pd.DataFrame(rows), 200, seed=3)
    assert sampled["reasoning_difficulty_bin"].value_counts().to_dict() == {
        "easy": 67,
        "medium": 67,
        "hard": 66,
    }


def test_prompt_requires_six_steps_and_forbids_invented_numbers():
    messages = MODULE.prompt_messages(object(), object(), "remove the sign")
    text = messages[0]["content"][2]["text"]
    assert "exactly six numbered steps" in text
    assert "Do not invent numerical measurements" in text
