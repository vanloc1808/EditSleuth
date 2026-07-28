#!/usr/bin/env python3
"""Download and materialize MagicBrush dev for release-based evaluation.

The output paths intentionally match ``magicbrush_dev_annotations.parquet``:

    <root>/dev/magicbrush_dev_<img_id>_t<turn>__real.png
    <root>/dev/magicbrush_dev_<img_id>_t<turn>__edited.png
    <root>/dev/magicbrush_dev_<img_id>_t<turn>__mask.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

from edit2forensics.utils.image_io import magicbrush_mask_to_binary


def _save_image(image: Image.Image, path: Path) -> None:
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def materialize_dev(records: Iterable[dict], root: Path) -> int:
    output = root / "dev"
    count = 0
    for index, record in enumerate(records):
        img_id = record.get("img_id")
        turn_index = record.get("turn_index")
        source = record.get("source_img")
        target = record.get("target_img")
        if img_id is None or turn_index is None or source is None or target is None:
            raise ValueError(f"MagicBrush dev row {index} is missing required data")

        triplet_id = f"magicbrush_dev_{img_id}_t{int(turn_index):02d}"
        _save_image(source.convert("RGB"), output / f"{triplet_id}__real.png")
        _save_image(target.convert("RGB"), output / f"{triplet_id}__edited.png")
        mask = record.get("mask_img")
        if mask is not None:
            _save_image(
                magicbrush_mask_to_binary(mask),
                output / f"{triplet_id}__mask.png",
            )
        count += 1
        if count % 50 == 0:
            print(f"Materialized {count} MagicBrush dev rows")
    return count


def validate_release_paths(annotations: Path, root: Path) -> None:
    df = pd.read_parquet(annotations, columns=["real_path", "edited_path"])
    missing = []
    for column in ("real_path", "edited_path"):
        for relative in df[column]:
            path = root / str(relative)
            if not path.exists():
                missing.append(str(path))
                if len(missing) == 10:
                    break
        if len(missing) == 10:
            break
    if missing:
        raise RuntimeError(
            "materialized MagicBrush paths do not match the release; examples:\n"
            + "\n".join(missing)
        )
    print(f"Validated all {len(df)} release rows against {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("editsleuth_data/release/magicbrush_dev_annotations.parquet"),
    )
    parser.add_argument("--hf-repo", default="osunlp/MagicBrush")
    parser.add_argument("--hf-cache-dir", type=Path)
    args = parser.parse_args()

    from datasets import load_dataset

    root = args.root.expanduser().resolve()
    print(f"Downloading {args.hf_repo} dev split")
    dataset = load_dataset(
        args.hf_repo,
        split="dev",
        cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
    )
    count = materialize_dev(dataset, root)
    print(f"Materialized {count} rows under {root / 'dev'}")
    validate_release_paths(args.annotations, root)
    print(f"MagicBrush dev ready at {root}")


if __name__ == "__main__":
    main()
