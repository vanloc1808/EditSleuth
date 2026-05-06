"""Tests for the pilot ``EditSleuthCurriculumDataset``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from edit2forensics.curriculum.dataset import (
    CurriculumDatasetConfig,
    EditSleuthCurriculumDataset,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> Path:
    arr = np.full((64, 64, 3), color, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _build_synthetic_corpus(tmp_path: Path, n_per_cell: int = 5):
    """Synthesize a tiny EditSleuth-shaped corpus on disk.

    Returns (triplets_path, reasoning_path) — both pointing at the
    parquet directories that the dataset will load.
    """
    triplets_path = tmp_path / "triplets.parquet"
    reasoning_path = tmp_path / "reasoning.parquet"
    triplets_path.mkdir()
    reasoning_path.mkdir()

    categories = ["object_addition", "object_removal", "style_transfer"]
    bins = ["easy", "medium", "hard"]

    triplet_rows = []
    reasoning_rows = []
    counter = 0
    for cat in categories:
        for bin_ in bins:
            for _ in range(n_per_cell):
                tid = f"t_{counter:04d}"
                counter += 1
                real_p = _write_image(tmp_path / f"{tid}_real.png", (200, 100, 100))
                edited_p = _write_image(tmp_path / f"{tid}_edited.png", (100, 200, 100))
                triplet_rows.append({
                    "triplet_id": tid,
                    "source_dataset": "test",
                    "real_path": str(real_p),
                    "edited_path": str(edited_p),
                    "instruction": f"do something for {tid}",
                    "provided_mask_path": None,
                    "metadata": json.dumps({}),
                })
                reasoning_rows.append({
                    "triplet_id": tid,
                    "header": f"[category={cat}, scope=local, "
                              f"difficulty={bin_}, source=dataset_label]",
                    "chain": (
                        f"1. The edit instruction states: \"do something for {tid}\".\n"
                        f"2. The mask of changed pixels covers roughly 5% of the image.\n"
                        f"3. Structural change is moderate.\n"
                        f"4. The edit is classified as {cat}.\n"
                        f"5. Edits of this type typically exhibit some signature.\n"
                        f"6. Overall, this triplet is of {bin_} detection difficulty."
                    ),
                    "template_version": "v1.0",
                    "category": cat,
                    "difficulty_bin": bin_,
                    "spatial_descriptor": "centered",
                })
    pd.DataFrame(triplet_rows).to_parquet(
        triplets_path / "part-00000.parquet", index=False,
    )
    pd.DataFrame(reasoning_rows).to_parquet(
        reasoning_path / "part-00000.parquet", index=False,
    )
    return triplets_path, reasoning_path


# ---------------------------------------------------------------------------
# Construction and shape
# ---------------------------------------------------------------------------

def test_dataset_loads_and_has_correct_size(tmp_path):
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=5)
    ds = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(samples_per_category_per_bin=3),
    )
    # 3 categories * 3 bins * 3 samples = 27 expected.
    assert len(ds) == 27


def test_dataset_caps_at_available_size(tmp_path):
    """If samples_per_category_per_bin exceeds available rows in a
    cell, we take all available without error."""
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=3)
    ds = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(samples_per_category_per_bin=10),
    )
    # 3 * 3 * min(10, 3) = 27 (capped at the cell size of 3).
    assert len(ds) == 27


def test_dataset_excludes_other_category(tmp_path):
    """The 'other' category is excluded from the pilot training set."""
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=5)
    # Append an 'other' row to the reasoning parquet.
    reasoning_df = pd.read_parquet(reasoning_path / "part-00000.parquet")
    triplets_df = pd.read_parquet(triplets_path / "part-00000.parquet")

    extra_tid = "t_other_001"
    real_p = _write_image(tmp_path / f"{extra_tid}_real.png", (50, 50, 50))
    edited_p = _write_image(tmp_path / f"{extra_tid}_edited.png", (60, 60, 60))
    triplets_df = pd.concat([triplets_df, pd.DataFrame([{
        "triplet_id": extra_tid, "source_dataset": "test",
        "real_path": str(real_p), "edited_path": str(edited_p),
        "instruction": "edit", "provided_mask_path": None,
        "metadata": json.dumps({}),
    }])], ignore_index=True)
    reasoning_df = pd.concat([reasoning_df, pd.DataFrame([{
        "triplet_id": extra_tid,
        "header": "[category=other, scope=local, difficulty=medium, source=fallback]",
        "chain": "1. ... 6. ...",
        "template_version": "v1.0",
        "category": "other",
        "difficulty_bin": "medium",
        "spatial_descriptor": "centered",
    }])], ignore_index=True)
    triplets_df.to_parquet(triplets_path / "part-00000.parquet", index=False)
    reasoning_df.to_parquet(reasoning_path / "part-00000.parquet", index=False)

    ds = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(samples_per_category_per_bin=5),
    )
    categories_in_ds = {r["category"] for r in ds.records}
    assert "other" not in categories_in_ds


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def test_dataset_stratifies_across_category_and_bin(tmp_path):
    """Stratified sampling should produce roughly equal counts per cell."""
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=5)
    ds = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(samples_per_category_per_bin=3),
    )
    counts: dict = {}
    for rec in ds.records:
        key = (rec["category"], rec["difficulty_bin"])
        counts[key] = counts.get(key, 0) + 1
    assert all(v == 3 for v in counts.values())


def test_dataset_seed_is_deterministic(tmp_path):
    """Two datasets built with the same seed must yield the same
    sampled records."""
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=5)
    cfg = CurriculumDatasetConfig(samples_per_category_per_bin=3, seed=42)
    ds_a = EditSleuthCurriculumDataset(triplets_path, reasoning_path, config=cfg)
    ds_b = EditSleuthCurriculumDataset(triplets_path, reasoning_path, config=cfg)
    ids_a = [r["triplet_id"] for r in ds_a.records]
    ids_b = [r["triplet_id"] for r in ds_b.records]
    assert ids_a == ids_b


def test_dataset_different_seeds_produce_different_samples(tmp_path):
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=5)
    ds_a = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(samples_per_category_per_bin=3, seed=0),
    )
    ds_b = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(samples_per_category_per_bin=3, seed=1),
    )
    ids_a = sorted(r["triplet_id"] for r in ds_a.records)
    ids_b = sorted(r["triplet_id"] for r in ds_b.records)
    assert ids_a != ids_b


# ---------------------------------------------------------------------------
# Item shape
# ---------------------------------------------------------------------------

def test_getitem_returns_chain_target_in_chain_mode(tmp_path):
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=2)
    ds = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(target_mode="chain",
                                       samples_per_category_per_bin=2),
    )
    item = ds[0]
    assert "real_image" in item and "edited_image" in item
    assert isinstance(item["target"], str)
    # Chain target contains the six numbered steps.
    for n in range(1, 7):
        assert f"{n}. " in item["target"]


def test_getitem_returns_label_only_target_in_label_only_mode(tmp_path):
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=2)
    ds = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(target_mode="label_only",
                                       samples_per_category_per_bin=2),
    )
    item = ds[0]
    # Label-only target is a JSON-serialized dict.
    parsed = json.loads(item["target"])
    assert "category" in parsed
    assert "spatial_descriptor" in parsed
    assert "difficulty_bin" in parsed
    assert parsed["category"] in ("object_addition", "object_removal", "style_transfer")


def test_image_resize_caps_at_max_side(tmp_path):
    """A high-resolution image should be downsampled to at most
    ``image_max_side`` on its longest dimension."""
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=2)
    # Overwrite the first triplet's images with high-res versions.
    rec = pd.read_parquet(triplets_path / "part-00000.parquet").iloc[0]
    big = np.full((1024, 1024, 3), 50, dtype=np.uint8)
    Image.fromarray(big).save(rec["real_path"])
    Image.fromarray(big).save(rec["edited_path"])

    ds = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(samples_per_category_per_bin=2,
                                       image_max_side=128),
    )
    # Find the resized record by triplet_id.
    matching = [i for i, r in enumerate(ds.records) if r["triplet_id"] == rec["triplet_id"]]
    assert matching, "expected the modified triplet to be present"
    item = ds[matching[0]]
    w, h = item["real_image"].size
    assert max(w, h) == 128


# ---------------------------------------------------------------------------
# HF-Datasets-like introspection surface
# ---------------------------------------------------------------------------

def test_dataset_exposes_column_names(tmp_path):
    """TRL's SFTTrainer (>= 0.11) probes ``dataset.column_names`` during
    pre-training setup. We expose the keys our ``__getitem__`` produces."""
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=2)
    ds = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(samples_per_category_per_bin=2),
    )
    cols = ds.column_names
    # Match the keys actually produced by __getitem__.
    item = ds[0]
    assert set(cols) == set(item.keys())
    # Required for VLM training: image fields, instruction, target.
    for required in ("real_image", "edited_image", "instruction", "target"):
        assert required in cols


def test_dataset_supports_in_membership_checks(tmp_path):
    """Some trainer code paths use ``"target" in dataset`` to detect
    whether a particular column is present."""
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=2)
    ds = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(samples_per_category_per_bin=2),
    )
    assert "target" in ds
    assert "real_image" in ds
    assert "nonexistent_column" not in ds


def test_column_names_are_consistent_across_target_modes(tmp_path):
    """The set of columns shouldn't depend on whether the target
    mode is ``chain`` or ``label_only`` — only the contents differ."""
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=2)
    ds_chain = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(target_mode="chain",
                                       samples_per_category_per_bin=2),
    )
    ds_label = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(target_mode="label_only",
                                       samples_per_category_per_bin=2),
    )
    assert ds_chain.column_names == ds_label.column_names




def test_missing_parquet_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no parquet shards"):
        EditSleuthCurriculumDataset(
            tmp_path / "nonexistent_triplets",
            tmp_path / "nonexistent_reasoning",
        )


def test_inner_join_drops_triplets_without_reasoning(tmp_path):
    """A triplet without a corresponding reasoning record is dropped
    (inner join), not silently included."""
    triplets_path, reasoning_path = _build_synthetic_corpus(tmp_path, n_per_cell=3)
    # Add an extra triplet without a matching reasoning row.
    extra_tid = "orphan_001"
    real_p = _write_image(tmp_path / f"{extra_tid}_real.png", (50, 50, 50))
    edited_p = _write_image(tmp_path / f"{extra_tid}_edited.png", (60, 60, 60))
    triplets_df = pd.read_parquet(triplets_path / "part-00000.parquet")
    triplets_df = pd.concat([triplets_df, pd.DataFrame([{
        "triplet_id": extra_tid, "source_dataset": "test",
        "real_path": str(real_p), "edited_path": str(edited_p),
        "instruction": "edit", "provided_mask_path": None,
        "metadata": json.dumps({}),
    }])], ignore_index=True)
    triplets_df.to_parquet(triplets_path / "part-00000.parquet", index=False)

    ds = EditSleuthCurriculumDataset(
        triplets_path, reasoning_path,
        config=CurriculumDatasetConfig(samples_per_category_per_bin=3),
    )
    triplet_ids = {r["triplet_id"] for r in ds.records}
    assert extra_tid not in triplet_ids
