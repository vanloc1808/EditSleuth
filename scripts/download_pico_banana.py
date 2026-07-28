#!/usr/bin/env python3
"""Download the Pico-Banana single-turn SFT data without Flickr requests.

The source Open Images archives are fetched from the public S3 bucket with
``aws s3 --no-sign-request``. Edited images and the SFT JSONL are fetched
from Apple's CDN. The resulting layout matches EditSleuth release paths:

    <root>/openimages/train_0/<image-id>.jpg
    <root>/openimages/train_1/<image-id>.jpg
    <root>/images/positive-edit/<edited-image>.png
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

SFT_MANIFEST_URL = (
    "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/"
    "nb/manifest/sft_manifest.txt"
)
SFT_JSONL_URL = (
    "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl"
)
OPENIMAGES_METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/train/"
    "train-images-boxable-with-rotation.csv"
)
OPENIMAGES_S3_PREFIX = "s3://open-images-dataset/tar"


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "EditSleuth/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    temporary.replace(destination)


def _run_aws_download(destination: Path, archive_name: str) -> Path:
    if shutil.which("aws") is None:
        raise RuntimeError("aws CLI is required but was not found on PATH")
    archive = destination / archive_name
    if archive.exists() and archive.stat().st_size:
        return archive
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aws", "s3", "--no-sign-request",
            "--endpoint-url", "https://s3.amazonaws.com",
            "cp", f"{OPENIMAGES_S3_PREFIX}/{archive_name}", str(archive),
        ],
        check=True,
    )
    return archive


def _extract_tar(archive: Path, output: Path) -> None:
    marker = output / f".{archive.name}.extracted"
    if marker.exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        base = output.resolve()
        for member in tar.getmembers():
            target = (output / member.name).resolve()
            if not target.is_relative_to(base):
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive links are not accepted: {member.name}")
        tar.extractall(output)
    marker.touch()


def _download_edited_images(manifest: Path, output: Path, workers: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    urls = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]

    def fetch(url: str) -> None:
        name = Path(urlparse(url).path).name
        if not name:
            raise ValueError(f"manifest URL has no filename: {url}")
        _download(url, output / name)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, urls))


def _write_local_manifest(
    source_jsonl: Path,
    metadata_csv: Path,
    openimages_root: Path,
    destination: Path,
) -> None:
    if destination.exists() and destination.stat().st_size:
        return
    url_to_id: dict[str, str] = {}
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            url_to_id[row["OriginalURL"].strip()] = row["ImageID"].strip()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with source_jsonl.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8",
    ) as output:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = url_to_id.get(record.get("open_image_input_url", ""))
            local_path = None
            if image_id:
                for split in ("train_0", "train_1"):
                    candidate = openimages_root / split / f"{image_id}.jpg"
                    if candidate.exists():
                        local_path = candidate.relative_to(openimages_root.parent)
                        break
            record["local_input_image"] = (
                local_path.as_posix() if local_path is not None else None
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    downloads = root / "downloads"
    manifests = root / "manifests"
    openimages = root / "openimages"

    sft_manifest = manifests / "sft_manifest.txt"
    sft_jsonl = manifests / "sft.jsonl"
    metadata_csv = manifests / "train-images-boxable-with-rotation.csv"
    _download(SFT_MANIFEST_URL, sft_manifest)
    _download(SFT_JSONL_URL, sft_jsonl)
    _download(OPENIMAGES_METADATA_URL, metadata_csv)

    archives = [
        _run_aws_download(downloads, f"train_{index}.tar.gz")
        for index in (0, 1)
    ]
    for archive in archives:
        _extract_tar(archive, openimages)

    _download_edited_images(
        sft_manifest, root / "images" / "positive-edit", args.workers,
    )
    _write_local_manifest(
        sft_jsonl,
        metadata_csv,
        openimages,
        root / "sft_with_local_source_image_path.jsonl",
    )

    if not args.keep_archives:
        for archive in archives:
            archive.unlink(missing_ok=True)

    print(f"Pico-Banana ready at {root}")


if __name__ == "__main__":
    main()
