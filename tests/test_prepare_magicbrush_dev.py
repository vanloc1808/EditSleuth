from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_magicbrush_dev.py"
SPEC = importlib.util.spec_from_file_location("prepare_magicbrush_dev", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_materialize_dev_matches_release_path_contract(tmp_path: Path):
    records = [{
        "img_id": "140513",
        "turn_index": 1,
        "source_img": Image.new("RGB", (4, 4), "white"),
        "target_img": Image.new("RGB", (4, 4), "red"),
        "mask_img": None,
    }]
    count = MODULE.materialize_dev(records, tmp_path)
    assert count == 1
    assert (
        tmp_path / "dev" / "magicbrush_dev_140513_t01__real.png"
    ).exists()
    assert (
        tmp_path / "dev" / "magicbrush_dev_140513_t01__edited.png"
    ).exists()
