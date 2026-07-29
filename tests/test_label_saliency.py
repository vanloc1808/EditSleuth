from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_label_saliency.py"
SPEC = importlib.util.spec_from_file_location("generate_label_saliency", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_normalize_saliency_handles_constant_map():
    result = MODULE.normalize_saliency(np.ones((3, 4)))
    assert np.array_equal(result, np.zeros((3, 4), dtype=np.float32))


def test_split_patch_saliency_respects_two_image_grids():
    scores = np.arange(10, dtype=np.float32)
    grids = np.array([[1, 2, 3], [1, 2, 2]])
    maps = MODULE.split_patch_saliency(scores, grids)
    assert [item.shape for item in maps] == [(2, 3), (2, 2)]
