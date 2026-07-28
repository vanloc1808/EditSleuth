from __future__ import annotations

from pathlib import Path

import pandas as pd

from edit2forensics.curriculum.dataset import (
    CurriculumDatasetConfig,
    EditSleuthCurriculumDataset,
)


def _release_row() -> dict:
    return {
        "triplet_id": "picobanana_1",
        "real_path": "openimages/train_0/source.jpg",
        "edited_path": "images/positive-edit/1.png",
        "instruction": "Replace the list with:\n1.) Buy food\n2.) Call Mom.",
        "reasoning_chain": (
            '1. The edit instruction states: "Replace the list with:\n'
            '1.) Buy food\n2.) Call Mom.".\n'
            "2. The mask covers 10% of the image.\n"
            "3. Structural change is minor.\n"
            "4. The edit is classified as object_removal.\n"
            "5. Edits of this type typically exhibit inpainting artifacts.\n"
            "6. Overall, this triplet is easier than average."
        ),
        "reasoning_category": "object_removal",
        "reasoning_difficulty_bin": "easy",
        "reasoning_spatial_descriptor": "centered",
    }


def test_release_schema_resolves_paths_and_redacts_chain_step_one(tmp_path: Path):
    annotations = tmp_path / "annotations.parquet"
    pd.DataFrame([_release_row()]).to_parquet(annotations)
    config = CurriculumDatasetConfig(
        samples_per_category_per_bin=1,
        include_instruction=False,
        redact_instruction_target=True,
    )
    dataset = EditSleuthCurriculumDataset(
        annotations_parquet=annotations,
        image_root=tmp_path / "pico",
        config=config,
    )
    record = dataset.records[0]
    assert record["real_path"] == str(
        (tmp_path / "pico" / "openimages/train_0/source.jpg").resolve()
    )
    target = dataset._build_target(record)
    assert target.startswith("1. The edit instruction is withheld for this ablation.")
    assert "Replace the list" not in target
    assert "Call Mom" not in target
    assert "2. The mask covers" in target


def test_label_only_target_is_unchanged_when_instruction_is_masked(tmp_path: Path):
    annotations = tmp_path / "annotations.parquet"
    pd.DataFrame([_release_row()]).to_parquet(annotations)
    config = CurriculumDatasetConfig(
        target_mode="label_only",
        samples_per_category_per_bin=1,
        include_instruction=False,
    )
    dataset = EditSleuthCurriculumDataset(
        annotations_parquet=annotations,
        image_root=tmp_path,
        config=config,
    )
    target = dataset._build_target(dataset.records[0])
    assert '"category": "object_removal"' in target
    assert '"difficulty_bin": "easy"' in target
