#!/usr/bin/env python3
"""Download and safely extract the public EditSleuth release from Drive."""
from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        base = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(base):
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive links are not accepted: {member.name}")
        tar.extractall(destination)


def _find_release_root(extracted: Path) -> Path:
    matches = list(extracted.rglob("pico_banana_annotations.parquet"))
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one pico_banana_annotations.parquet in archive, "
            f"found {len(matches)}"
        )
    root = matches[0].parent
    if not (root / "magicbrush_dev_annotations.parquet").exists():
        raise RuntimeError("archive is missing magicbrush_dev_annotations.parquet")
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", required=True)
    parser.add_argument(
        "--archive", type=Path, default=Path("editsleuth_data.tar.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("editsleuth_data/release"),
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("install gdown first: pip install gdown") from exc

    archive = args.archive.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    if args.force_download or not archive.exists():
        result = gdown.download(id=args.file_id, output=str(archive), quiet=False)
        if result is None:
            raise RuntimeError("gdown did not produce the requested archive")

    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"{output} is not empty; move it aside before extracting to avoid "
            "mixing release versions"
        )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="editsleuth-release-", dir=output.parent,
    ) as temporary:
        temporary_path = Path(temporary)
        _safe_extract(archive, temporary_path)
        release_root = _find_release_root(temporary_path)
        for child in release_root.iterdir():
            shutil.move(str(child), output / child.name)
    print(f"EditSleuth release ready at {output}")


if __name__ == "__main__":
    main()
