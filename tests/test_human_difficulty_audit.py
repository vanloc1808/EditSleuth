from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_human_difficulty.py"
SPEC = importlib.util.spec_from_file_location("audit_human_difficulty", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_balanced_sample_has_requested_size_and_all_cells():
    rows = []
    for bin_name in MODULE.LEVELS:
        for category in ("a", "b"):
            for index in range(20):
                rows.append({
                    "reasoning_difficulty_bin": bin_name,
                    "category_category": category,
                    "row": f"{bin_name}-{category}-{index}",
                })
    selected = MODULE.balanced_sample(pd.DataFrame(rows), 60, seed=7)
    assert len(selected) == 60
    assert set(selected["reasoning_difficulty_bin"]) == set(MODULE.LEVELS)
    assert set(selected["category_category"]) == {"a", "b"}


def test_quadratic_kappa_is_one_for_identical_labels():
    labels = np.array([1, 2, 3, 1, 3])
    assert MODULE.quadratic_weighted_kappa(labels, labels) == 1.0
