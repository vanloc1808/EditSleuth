"""Pilot curriculum dataset for EditSleuth fine-tuning.

Builds training examples from the Stage E reasoning parquet plus
upstream Stage A artifacts (the (real, edited) image pair). Each
example consists of the two source images, the edit instruction,
and a target string that is either the full reasoning chain
(``chain`` target mode) or a structured label triple (``label_only``
target mode).

Pilot scope: stratified random sampling balanced by category and
difficulty bin. No easy-to-hard scheduling; that's left for full-
scale curriculum work in a follow-up paper.

The dataset returns dicts compatible with HuggingFace's chat-template
processors for vision-language models (Qwen2-VL specifically; the
schema is general enough for other VL models with minor adaptation).
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from PIL import Image

# Soft dependency on torch: the dataset class is a plain Python class
# (with __len__ and __getitem__) and does not need to inherit from
# torch.utils.data.Dataset to be usable by torch's DataLoader. Importing
# torch eagerly would require it as a hard dependency of the whole
# codebase, which we avoid for the non-pilot pipeline.
try:
    from torch.utils.data import Dataset as _TorchDataset
    _BASE: type = _TorchDataset
except ImportError:  # pragma: no cover — torch may be absent in CI
    _BASE = object

log = logging.getLogger(__name__)

TargetMode = Literal["chain", "label_only"]


@dataclass
class CurriculumDatasetConfig:
    """Pilot dataset configuration."""

    target_mode: TargetMode = "chain"
    """Whether the supervised target is the full reasoning chain or
    a structured label-only triple."""

    samples_per_category_per_bin: int = 600
    """Stratified-sample size: each (category, difficulty_bin) cell
    contributes this many triplets to the training set. With 10
    Pico-Banana categories and 3 bins, the default produces
    18K training examples."""

    seed: int = 0
    """Random seed for stratified sampling. Held fixed so the pilot
    is reproducible."""

    image_max_side: int = 448
    """Long-side resize for images before feeding to the VLM. 448 is
    a typical input resolution for Qwen2-VL-2B."""

    include_instruction: bool = True
    """Whether the edit instruction is exposed in the user prompt."""

    redact_instruction_target: bool = True
    """When the instruction is withheld, replace chain Step 1 with a
    fixed redaction marker so the model is not trained to reconstruct
    hidden text."""


class EditSleuthCurriculumDataset(_BASE):
    """Pilot training dataset.

    The dataset is materialized in memory as a list of records
    (one per training example) once at construction time. This is
    fine at pilot scale (~20K samples) but would need a streaming
    rewrite for the full 257K corpus.

    Compatibility note
    ------------------
    Recent versions of TRL's ``SFTTrainer`` (>= 0.11) probe a
    dataset's ``column_names`` attribute during pre-training setup,
    even when ``remove_unused_columns=False`` is set in the training
    arguments. We expose ``column_names`` as a property returning the
    keys our ``__getitem__`` produces, so the trainer's introspection
    doesn't fail with ``AttributeError``. The other HF-Datasets-like
    surface (``features``, ``__contains__``) is also exposed where
    cheap to provide.
    """

    # Fixed list of keys that ``__getitem__`` produces. Kept as a
    # class attribute so the property is cheap and deterministic.
    _COLUMN_NAMES = (
        "triplet_id",
        "real_image",
        "edited_image",
        "instruction",
        "target",
        "category",
        "difficulty_bin",
    )

    def __init__(
        self,
        triplets_parquet: Path | None = None,
        reasoning_parquet: Path | None = None,
        config: CurriculumDatasetConfig | None = None,
        *,
        annotations_parquet: Path | None = None,
        image_root: Path | None = None,
    ) -> None:
        self.config = config or CurriculumDatasetConfig()

        if annotations_parquet is not None:
            log.info("loading joined release annotations: %s", annotations_parquet)
            df = pd.read_parquet(annotations_parquet)
            df = self._normalize_release_annotations(df, image_root)
        else:
            if triplets_parquet is None or reasoning_parquet is None:
                raise ValueError(
                    "provide annotations_parquet, or both triplets_parquet "
                    "and reasoning_parquet"
                )
            log.info("loading triplets and reasoning artifacts")
            triplets_df = self._load_parquet_dataset(triplets_parquet)
            reasoning_df = self._load_parquet_dataset(reasoning_parquet)
            df = triplets_df.merge(
                reasoning_df, on="triplet_id", how="inner", suffixes=("", "_r"),
            )
        log.info("joined: %d triplets with reasoning artifacts", len(df))

        # Drop "other" — pilot doesn't include the unclassified residual.
        df = df[df["category"] != "other"]

        # Stratified sample: ``samples_per_category_per_bin`` from each
        # (category, difficulty_bin) cell.
        rng = random.Random(self.config.seed)
        sampled_groups = []
        for (cat, bin_), group in df.groupby(["category", "difficulty_bin"]):
            n_take = min(self.config.samples_per_category_per_bin, len(group))
            sampled_idx = rng.sample(range(len(group)), n_take)
            sampled_groups.append(group.iloc[sampled_idx])
        sampled = pd.concat(sampled_groups, ignore_index=True)
        # Shuffle the final mix so within an epoch the model sees
        # interleaved categories and difficulties.
        sampled = sampled.sample(
            frac=1, random_state=self.config.seed,
        ).reset_index(drop=True)
        log.info(
            "sampled %d training records (target_mode=%s)",
            len(sampled), self.config.target_mode,
        )
        self.records = sampled.to_dict("records")

    @staticmethod
    def _normalize_release_annotations(
        df: pd.DataFrame,
        image_root: Path | None,
    ) -> pd.DataFrame:
        """Map the public joined release schema to the pilot schema."""
        required = {
            "triplet_id", "real_path", "edited_path", "instruction",
            "reasoning_chain", "reasoning_category",
            "reasoning_difficulty_bin", "reasoning_spatial_descriptor",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"release annotations missing required columns: {sorted(missing)}"
            )
        if image_root is None:
            raise ValueError(
                "image_root is required with annotations_parquet because "
                "release image paths are relative"
            )
        root = Path(image_root).expanduser().resolve()
        out = df.copy()
        out["real_path"] = out["real_path"].map(
            lambda value: str(root / str(value))
        )
        out["edited_path"] = out["edited_path"].map(
            lambda value: str(root / str(value))
        )
        out["chain"] = out["reasoning_chain"]
        out["category"] = out["reasoning_category"]
        out["difficulty_bin"] = out["reasoning_difficulty_bin"]
        out["spatial_descriptor"] = out["reasoning_spatial_descriptor"]
        return out

    @staticmethod
    def _load_parquet_dataset(path: Path) -> pd.DataFrame:
        shards = sorted(path.glob("part-*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no parquet shards under {path}")
        return pd.concat(
            [pd.read_parquet(s) for s in shards], ignore_index=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------ #
    # HF-Datasets-like introspection surface
    # ------------------------------------------------------------------ #
    # These let TRL's ``SFTTrainer`` and similar trainers probe the
    # dataset without expecting a real ``datasets.Dataset`` object.

    @property
    def column_names(self) -> list[str]:
        """Names of the keys ``__getitem__`` produces. Required by
        TRL >= 0.11 during pre-training setup."""
        return list(self._COLUMN_NAMES)

    def __contains__(self, key: str) -> bool:
        """Allow ``"target" in dataset`` style checks that some
        trainer code paths use to detect whether a particular column
        is present."""
        return key in self._COLUMN_NAMES

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        real_img = self._load_image(rec["real_path"])
        edited_img = self._load_image(rec["edited_path"])

        instruction = rec.get("instruction", "") or ""
        target = self._build_target(rec)

        return {
            "triplet_id": rec["triplet_id"],
            "real_image": real_img,
            "edited_image": edited_img,
            "instruction": instruction,
            "target": target,
            # Mirrored fields useful for debugging / weighting.
            "category": rec["category"],
            "difficulty_bin": rec["difficulty_bin"],
        }

    def _load_image(self, path_str: str) -> Image.Image:
        img = Image.open(path_str).convert("RGB")
        # Long-side resize to keep VRAM bounded.
        max_side = self.config.image_max_side
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize(
                (int(w * scale), int(h * scale)), Image.BILINEAR,
            )
        return img

    def _build_target(self, rec: dict[str, Any]) -> str:
        """Compose the target string.

        Chain mode: return the full Stage E ``chain`` field
        verbatim. The model learns to reproduce the chain given
        the (real, edited, instruction) input.

        Label-only mode: serialize a structured triple of
        (category, scope, difficulty_bin) as a JSON string. The
        model learns to predict these labels without per-step
        reasoning.
        """
        if self.config.target_mode == "chain":
            chain = rec["chain"]
            if (
                not self.config.include_instruction
                and self.config.redact_instruction_target
            ):
                chain, replacements = re.subn(
                    r"\A1\. .*?(?=^2\. )",
                    "1. The edit instruction is withheld for this ablation.\n",
                    chain,
                    count=1,
                    flags=re.DOTALL | re.MULTILINE,
                )
                if replacements != 1:
                    raise ValueError(
                        f"cannot redact malformed chain for {rec['triplet_id']}"
                    )
            return chain
        # label_only
        return json.dumps({
            "category": rec["category"],
            "spatial_descriptor": rec.get("spatial_descriptor", "centered"),
            "difficulty_bin": rec["difficulty_bin"],
        })
