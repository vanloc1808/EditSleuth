"""Tests for the MagicBrush adapter.

We do NOT hit HuggingFace in tests. Instead we monkeypatch
`datasets.load_dataset` to return an in-memory list of dicts that mimic
the MagicBrush schema. This keeps the suite fast and network-free while
still exercising the full adapter path: record parsing, image
materialization, ID construction, multi-turn handling, and metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from edit2forensics.adapters.magicbrush import MagicBrushAdapter
from edit2forensics.data.triplet import EditTriplet


def _img(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _mask(size: tuple[int, int] = (32, 32)) -> Image.Image:
    """A simple L-mode mask with a white square in the top-left quadrant."""
    m = Image.new("L", size, color=0)
    for x in range(size[0] // 2):
        for y in range(size[1] // 2):
            m.putpixel((x, y), 255)
    return m


def _fake_records() -> list[dict[str, Any]]:
    """Three synthetic rows: a turn-1, a turn-2 (same img_id), and another turn-1."""
    return [
        {
            "img_id": "000000391895",
            "turn_index": 1,
            "source_img": _img((255, 0, 0)),
            "mask_img": _mask(),
            "instruction": "Add a hat to the man.",
            "target_img": _img((200, 0, 0)),
        },
        {
            "img_id": "000000391895",
            "turn_index": 2,
            "source_img": _img((200, 0, 0)),  # previous turn's output
            "mask_img": _mask(),
            "instruction": "Change the hat color to blue.",
            "target_img": _img((200, 50, 50)),
        },
        {
            "img_id": "000000522418",
            "turn_index": 1,
            "source_img": _img((0, 255, 0)),
            "mask_img": _mask(),
            "instruction": "Replace the dog with a cat.",
            "target_img": _img((0, 200, 0)),
        },
    ]


@pytest.fixture
def patch_load_dataset(monkeypatch):
    """Patch `datasets.load_dataset` inside the adapter module.

    We patch at the import site (`datasets.load_dataset`) rather than on
    some pre-imported symbol, since the adapter lazy-imports the function
    inside `ingest()`.
    """
    import datasets as hf_datasets

    def fake_load_dataset(repo, split=None, cache_dir=None):
        assert repo == "osunlp/MagicBrush"
        assert split in ("train", "dev")
        return _fake_records()

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    yield


def test_adapter_basic_ingest(tmp_path, patch_load_dataset):
    adapter = MagicBrushAdapter(cache_root=tmp_path, split="train")
    triplets = list(adapter.ingest())

    assert len(triplets) == 3

    # --- IDs encode split, img_id, and turn_index ---
    ids = [t.triplet_id for t in triplets]
    assert ids == [
        "magicbrush_train_000000391895_t01",
        "magicbrush_train_000000391895_t02",
        "magicbrush_train_000000522418_t01",
    ]

    # --- source_dataset is consistent across the batch ---
    assert all(t.source_dataset == "magicbrush" for t in triplets)

    # --- paths are absolute and files exist on disk ---
    for t in triplets:
        assert t.real_path.is_absolute() and t.real_path.exists()
        assert t.edited_path.is_absolute() and t.edited_path.exists()
        assert t.provided_mask_path is not None
        assert t.provided_mask_path.exists()

    # --- mask is persisted in grayscale mode ---
    m = Image.open(triplets[0].provided_mask_path)
    assert m.mode == "L"

    # --- instruction is preserved verbatim ---
    assert triplets[0].instruction == "Add a hat to the man."


def test_metadata_flags_authentic_only_for_turn_one(tmp_path, patch_load_dataset):
    adapter = MagicBrushAdapter(cache_root=tmp_path, split="train")
    triplets = list(adapter.ingest())
    t1, t2, t3 = triplets

    assert t1.metadata["source_is_authentic"] is True    # turn 1
    assert t2.metadata["source_is_authentic"] is False   # turn 2
    assert t3.metadata["source_is_authentic"] is True    # turn 1

    # Other metadata fields are preserved
    assert t1.metadata["img_id"] == "000000391895"
    assert t1.metadata["turn_index"] == 1
    assert t1.metadata["split"] == "train"


def test_single_turn_only_filter(tmp_path, patch_load_dataset):
    adapter = MagicBrushAdapter(
        cache_root=tmp_path, split="train", single_turn_only=True
    )
    triplets = list(adapter.ingest())
    assert len(triplets) == 2
    assert all(t.metadata["turn_index"] == 1 for t in triplets)


def test_materialization_is_idempotent(tmp_path, patch_load_dataset):
    """Running the adapter twice should not duplicate on-disk files."""
    adapter1 = MagicBrushAdapter(cache_root=tmp_path, split="train")
    triplets1 = list(adapter1.ingest())
    # Capture mtimes from first run
    mtimes1 = {t.real_path: t.real_path.stat().st_mtime_ns for t in triplets1}

    adapter2 = MagicBrushAdapter(cache_root=tmp_path, split="train")
    triplets2 = list(adapter2.ingest())

    # Same IDs, same paths, same mtimes (skip_if_exists=True preserves files)
    assert [t.triplet_id for t in triplets1] == [t.triplet_id for t in triplets2]
    for t in triplets2:
        assert t.real_path.stat().st_mtime_ns == mtimes1[t.real_path]


def test_split_validation(tmp_path):
    with pytest.raises(ValueError, match="split must be"):
        MagicBrushAdapter(cache_root=tmp_path, split="test")


def test_split_appears_in_triplet_id_and_cache_path(tmp_path, patch_load_dataset):
    """Dev and train must never collide in IDs or on-disk layout."""
    adapter_train = MagicBrushAdapter(cache_root=tmp_path, split="train")
    adapter_dev = MagicBrushAdapter(cache_root=tmp_path, split="dev")

    t_train = list(adapter_train.ingest())[0]
    t_dev = list(adapter_dev.ingest())[0]

    assert "train" in t_train.triplet_id
    assert "dev" in t_dev.triplet_id
    assert t_train.triplet_id != t_dev.triplet_id
    assert t_train.real_path != t_dev.real_path


def test_skips_row_missing_required_fields(tmp_path, monkeypatch, caplog):
    """A record missing img_id / turn_index / instruction is skipped."""
    import datasets as hf_datasets

    records = [
        {"img_id": "", "turn_index": 1, "source_img": _img((1, 1, 1)),
         "mask_img": _mask(), "instruction": "x", "target_img": _img((2, 2, 2))},
        {"img_id": "abc", "turn_index": 1, "source_img": None,
         "mask_img": _mask(), "instruction": "x", "target_img": _img((2, 2, 2))},
        # Valid control record
        {"img_id": "ok", "turn_index": 1, "source_img": _img((3, 3, 3)),
         "mask_img": _mask(), "instruction": "x", "target_img": _img((4, 4, 4))},
    ]
    monkeypatch.setattr(
        hf_datasets, "load_dataset",
        lambda repo, split=None, cache_dir=None: records,
    )

    adapter = MagicBrushAdapter(cache_root=tmp_path, split="train")
    with caplog.at_level("WARNING"):
        triplets = list(adapter.ingest())
    assert len(triplets) == 1
    assert triplets[0].metadata["img_id"] == "ok"
    assert any("missing required field" in r.message for r in caplog.records) \
        or any("missing source or target" in r.message for r in caplog.records)


def test_mask_absent_is_tolerated(tmp_path, monkeypatch):
    """If mask_img is None, provided_mask_path should be None, not error."""
    import datasets as hf_datasets
    records = [
        {"img_id": "x", "turn_index": 1, "source_img": _img((1, 1, 1)),
         "mask_img": None, "instruction": "do a thing",
         "target_img": _img((2, 2, 2))},
    ]
    monkeypatch.setattr(
        hf_datasets, "load_dataset",
        lambda repo, split=None, cache_dir=None: records,
    )

    adapter = MagicBrushAdapter(cache_root=tmp_path, split="train")
    triplets = list(adapter.ingest())
    assert len(triplets) == 1
    assert triplets[0].provided_mask_path is None
    # Real and edited images still get materialized
    assert triplets[0].real_path.exists()
    assert triplets[0].edited_path.exists()


def test_max_records_cap(tmp_path, patch_load_dataset):
    adapter = MagicBrushAdapter(cache_root=tmp_path, split="train", max_records=2)
    triplets = list(adapter.ingest())
    assert len(triplets) == 2


def test_triplet_serialization_roundtrip(tmp_path, patch_load_dataset):
    adapter = MagicBrushAdapter(cache_root=tmp_path, split="train")
    original = next(adapter.ingest())
    restored = EditTriplet.from_dict(original.to_dict())
    assert restored == original
    # Especially: mask path survives the roundtrip
    assert restored.provided_mask_path == original.provided_mask_path
