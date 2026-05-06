"""Tests for the Pico-Banana adapter.

We build a tiny synthetic dataset on disk (manifest + dummy image files)
and verify the adapter produces `EditTriplet`s with the expected fields.

These tests do not hit the network. Download-fallback paths are tested
by pointing at a URL that's expected to fail and asserting the record
is skipped rather than raising.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from edit2forensics.adapters.pico_banana import PicoBananaAdapter
from edit2forensics.data.triplet import EditTriplet


def _make_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    """Write a 1-pixel-ish PNG/JPEG so `.exists()` and opens succeed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(128, 128, 128))
    img.save(path)


@pytest.fixture
def pico_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Build a synthetic Pico-Banana layout and return (manifest_path, root)."""
    root = tmp_path / "pico-banana"
    root.mkdir()

    # Create two dummy "real" images under openimages/...
    real1 = root / "openimages/images/train_0/aaaa.jpg"
    real2 = root / "openimages/images/train_0/bbbb.jpg"
    _make_image(real1)
    _make_image(real2)

    # Create two dummy "edited" images under images/positive-edit/
    edited1 = root / "images/positive-edit/1.png"
    edited2 = root / "images/positive-edit/42.png"
    _make_image(edited1)
    _make_image(edited2)

    # Write the JSONL manifest
    manifest = root / "pico_banana_400k.jsonl"
    records = [
        {
            "open_image_input_url": "https://example.invalid/a.jpg",
            "text": "Remove the red flag and extend the sky.",
            "output_image": "images/positive-edit/1.png",
            "edit_type": "Remove an existing object",
            "summarized_text": "Flag removed.",
            "local_input_image": "openimages/images/train_0/aaaa.jpg",
        },
        {
            "open_image_input_url": "https://example.invalid/b.jpg",
            "text": "Change the sky color to sunset orange.",
            "output_image": "images/positive-edit/42.png",
            "edit_type": "Color change",
            "summarized_text": "Sky -> sunset.",
            "local_input_image": "openimages/images/train_0/bbbb.jpg",
        },
    ]
    with open(manifest, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return manifest, root


def test_adapter_basic_ingest(pico_dataset):
    manifest, root = pico_dataset
    adapter = PicoBananaAdapter(manifest_path=manifest, dataset_root=root)
    triplets = list(adapter.ingest())

    assert len(triplets) == 2
    t1, t2 = triplets

    # --- IDs are derived from the filename stem of `output_image`, stable ---
    assert t1.triplet_id == "picobanana_1"
    assert t2.triplet_id == "picobanana_42"

    # --- source_dataset is set correctly ---
    assert t1.source_dataset == "pico-banana"

    # --- paths are absolute and point to existing files ---
    assert t1.real_path.is_absolute()
    assert t1.edited_path.is_absolute()
    assert t1.real_path.exists()
    assert t1.edited_path.exists()

    # --- instruction is the long-form `text` ---
    assert t1.instruction == "Remove the red flag and extend the sky."

    # --- Pico-Banana has no masks ---
    assert t1.provided_mask_path is None

    # --- metadata preserves source fields ---
    assert t1.metadata["source_edit_type"] == "Remove an existing object"
    assert t1.metadata["summarized_text"] == "Flag removed."
    assert t1.metadata["manifest_line_no"] == 1
    assert t1.metadata["source_output_image"] == "images/positive-edit/1.png"


def test_skip_missing_edited_image(pico_dataset, tmp_path, caplog):
    """An output_image that doesn't exist on disk must cause a skip."""
    manifest, root = pico_dataset
    # Append a record whose edited image does NOT exist.
    with open(manifest, "a") as f:
        f.write(json.dumps({
            "open_image_input_url": "https://example.invalid/c.jpg",
            "text": "Does not matter.",
            "output_image": "images/positive-edit/9999.png",   # missing
            "edit_type": "x",
            "summarized_text": "x",
            "local_input_image": "openimages/images/train_0/aaaa.jpg",
        }) + "\n")

    adapter = PicoBananaAdapter(manifest_path=manifest, dataset_root=root)
    with caplog.at_level("WARNING"):
        triplets = list(adapter.ingest())
    assert len(triplets) == 2  # the bad record is dropped
    assert any("edited image missing" in r.message for r in caplog.records)


def test_skip_missing_real_without_download(pico_dataset, caplog):
    """When local_input_image is missing and download_missing=False, skip."""
    manifest, root = pico_dataset
    # Use a NEW output_image path so this record doesn't collide with the
    # base fixture's first record. We're testing the missing-real skip,
    # not output_image collision detection.
    extra_edited = root / "images/positive-edit/777.png"
    Image.new("RGB", (32, 32), color=(64, 64, 64)).save(extra_edited)
    with open(manifest, "a") as f:
        f.write(json.dumps({
            "open_image_input_url": "https://example.invalid/c.jpg",
            "text": "Some edit.",
            "output_image": "images/positive-edit/777.png",
            "edit_type": "x",
            "summarized_text": "x",
            "local_input_image": "openimages/images/train_0/MISSING.jpg",
        }) + "\n")

    adapter = PicoBananaAdapter(
        manifest_path=manifest, dataset_root=root, download_missing=False
    )
    with caplog.at_level("WARNING"):
        triplets = list(adapter.ingest())
    # original 2 + the new record with missing real -> still only 2 yielded
    assert len(triplets) == 2
    assert any(
        "real image missing and download_missing=False" in r.message
        for r in caplog.records
    )


def test_malformed_json_line_is_skipped(pico_dataset, caplog):
    manifest, root = pico_dataset
    with open(manifest, "a") as f:
        f.write("this is not json\n")
    adapter = PicoBananaAdapter(manifest_path=manifest, dataset_root=root)
    with caplog.at_level("WARNING"):
        triplets = list(adapter.ingest())
    assert len(triplets) == 2
    assert any("malformed JSON" in r.message for r in caplog.records)


def test_max_records_cap(pico_dataset):
    manifest, root = pico_dataset
    adapter = PicoBananaAdapter(
        manifest_path=manifest, dataset_root=root, max_records=1
    )
    triplets = list(adapter.ingest())
    assert len(triplets) == 1
    assert triplets[0].triplet_id == "picobanana_1"


def test_triplet_serialization_roundtrip(pico_dataset):
    """EditTriplet.to_dict / from_dict should be lossless."""
    manifest, root = pico_dataset
    adapter = PicoBananaAdapter(manifest_path=manifest, dataset_root=root)
    original = next(adapter.ingest())
    restored = EditTriplet.from_dict(original.to_dict())
    assert restored == original


def test_string_filename_does_not_collide_with_numeric_filename(tmp_path):
    """Regression test for an ID-collision bug observed on the full
    Pico-Banana release.

    Some Pico-Banana entries have non-numeric filenames such as
    ``kewsee_retry1.png``. The earlier integer-extracting parser
    pulled out only the trailing digit (``1``), zero-padded it
    (``00000001``), and produced the same triplet_id as a record with
    ``output_image=1.png``. ~22% of triplet_ids on the full release
    collided as a result.

    This test creates a manifest with both filename styles and asserts
    that the resulting IDs are distinct.
    """
    root = tmp_path / "pico_collision_repro"
    root.mkdir()

    # Three edited images: one numeric, two with the trailing-digit
    # pattern that broke the old parser.
    edited_files = {
        "images/positive-edit/1.png":               "real_a.jpg",
        "images/positive-edit/kewsee_retry1.png":   "real_b.jpg",
        "images/positive-edit/foo_retry1.png":      "real_c.jpg",
    }
    for edited_rel, real_basename in edited_files.items():
        _make_image(root / edited_rel)
        _make_image(root / "openimages/images/train_0" / real_basename)

    manifest = root / "pico_banana_400k.jsonl"
    with open(manifest, "w") as f:
        for edited_rel, real_basename in edited_files.items():
            f.write(json.dumps({
                "open_image_input_url": "https://example.invalid/x.jpg",
                "text": f"Edit for {edited_rel}.",
                "output_image": edited_rel,
                "edit_type": "fixture",
                "summarized_text": "fixture",
                "local_input_image": f"openimages/images/train_0/{real_basename}",
            }) + "\n")

    adapter = PicoBananaAdapter(manifest_path=manifest, dataset_root=root)
    triplets = list(adapter.ingest())

    assert len(triplets) == 3
    ids = {t.triplet_id for t in triplets}
    # All three IDs distinct — no collision.
    assert len(ids) == 3
    assert ids == {
        "picobanana_1",
        "picobanana_kewsee_retry1",
        "picobanana_foo_retry1",
    }


# ---------------------------------------------------------------------------
# output_image collision handling
# ---------------------------------------------------------------------------

def _build_collision_manifest(root: Path) -> Path:
    """Build a manifest with two output_image collisions plus three clean
    records, exactly mirroring the pattern observed in the real
    Pico-Banana data (lines 177/382 collide on `463.png`, lines 679/842
    collide on `1021.png`)."""
    # Five records:
    #   collision A: output_image=463.png at lines 1 and 3
    #   collision B: output_image=1021.png at lines 2 and 5
    #   clean:       output_image=99.png at line 4
    edited_files = ["463.png", "1021.png", "463.png", "99.png", "1021.png"]
    for f in set(edited_files):
        _make_image(root / "images/positive-edit" / f)
    real_files = [f"real_{i}.jpg" for i in range(5)]
    for r in real_files:
        _make_image(root / "openimages/images/train_0" / r)

    manifest = root / "pico_banana_400k.jsonl"
    with open(manifest, "w") as f:
        for i, (out, real) in enumerate(zip(edited_files, real_files)):
            f.write(json.dumps({
                "open_image_input_url": f"https://example.invalid/{i}.jpg",
                "text": f"Edit number {i}.",
                "output_image": f"images/positive-edit/{out}",
                "edit_type": "fixture",
                "summarized_text": "fixture",
                "local_input_image": f"openimages/images/train_0/{real}",
            }) + "\n")
    return manifest


def test_output_collision_drop_all_default(tmp_path, caplog):
    """Default policy: drop every record involved in an output_image
    collision. From 5 records (2 collisions + 1 clean), only the 1
    clean record survives."""
    root = tmp_path / "pico_collision"
    root.mkdir()
    manifest = _build_collision_manifest(root)

    adapter = PicoBananaAdapter(manifest_path=manifest, dataset_root=root)
    with caplog.at_level("WARNING"):
        triplets = list(adapter.ingest())

    # One clean record survives.
    assert len(triplets) == 1
    assert triplets[0].triplet_id == "picobanana_99"

    # Collisions were surfaced in the log.
    assert any("collisions involving" in r.message for r in caplog.records)
    assert any("collision on id stem" in r.message for r in caplog.records)


def test_output_collision_keep_first_policy(tmp_path):
    """`keep_first` policy: keep the first manifest occurrence of each
    collision; drop the rest. From 5 records (2 collisions + 1 clean),
    we keep 3 records — the first of each collision plus the clean one.
    """
    root = tmp_path / "pico_collision_kf"
    root.mkdir()
    manifest = _build_collision_manifest(root)

    adapter = PicoBananaAdapter(
        manifest_path=manifest, dataset_root=root,
        on_output_collision="keep_first",
    )
    triplets = list(adapter.ingest())

    # Three records: first collision-A occurrence (line 1, id 463),
    # first collision-B occurrence (line 2, id 1021), and the clean
    # record (line 4, id 99).
    assert len(triplets) == 3
    ids = sorted(t.triplet_id for t in triplets)
    assert ids == ["picobanana_1021", "picobanana_463", "picobanana_99"]


def test_no_output_collision_yields_all_records(tmp_path):
    """A clean manifest without collisions should be unaffected by the
    new logic — all records yielded."""
    root = tmp_path / "pico_clean"
    root.mkdir()
    edited_files = ["1.png", "2.png", "3.png"]
    for f in edited_files:
        _make_image(root / "images/positive-edit" / f)
    for i in range(3):
        _make_image(root / f"openimages/images/train_0/real_{i}.jpg")
    manifest = root / "pico_banana_400k.jsonl"
    with open(manifest, "w") as f:
        for i, out in enumerate(edited_files):
            f.write(json.dumps({
                "open_image_input_url": "https://example.invalid/x.jpg",
                "text": f"Edit {i}.",
                "output_image": f"images/positive-edit/{out}",
                "edit_type": "fixture",
                "summarized_text": "fixture",
                "local_input_image": f"openimages/images/train_0/real_{i}.jpg",
            }) + "\n")

    adapter = PicoBananaAdapter(manifest_path=manifest, dataset_root=root)
    triplets = list(adapter.ingest())
    assert len(triplets) == 3


def test_invalid_collision_policy_rejected(tmp_path):
    """Constructor must validate the policy string at instantiation
    time, not at ingest time."""
    root = tmp_path / "pico_x"
    root.mkdir()
    manifest = root / "manifest.jsonl"
    manifest.touch()
    with pytest.raises(ValueError, match="on_output_collision"):
        PicoBananaAdapter(
            manifest_path=manifest, dataset_root=root,
            on_output_collision="drop_first_only",
        )


def test_requires_existing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        PicoBananaAdapter(
            manifest_path=tmp_path / "nope.jsonl",
            dataset_root=tmp_path,
        )


def test_download_missing_requires_cache(pico_dataset):
    manifest, root = pico_dataset
    with pytest.raises(ValueError, match="download_cache"):
        PicoBananaAdapter(
            manifest_path=manifest,
            dataset_root=root,
            download_missing=True,
            download_cache=None,
        )


def test_id_stem_derivation_edge_cases():
    """Directly exercise the id-stem derivation on tricky filenames.

    This replaces an earlier integer-only regex test. The integer-only
    behavior was a bug: filenames like ``kewsee_retry1.png`` collided
    with ``1.png``. The new behavior preserves the full stem.
    """
    d = PicoBananaAdapter._derive_id_stem
    # Numeric stems are preserved verbatim (no zero-padding anymore).
    assert d("images/positive-edit/1.png") == "1"
    assert d("images/positive-edit/00042.png") == "00042"
    assert d("a/b/c/123456.jpg") == "123456"
    # Non-numeric stems are preserved.
    assert d("images/positive-edit/kewsee_retry1.png") == "kewsee_retry1"
    assert d("a/b/c/foo_bar.png") == "foo_bar"
    # Directory components are ignored — only the basename's stem matters.
    assert d("folder_123/abc/7.png") == "7"
    assert d("with_99_in_path/edits/foo.png") == "foo"
    # Unsafe characters are replaced with underscores.
    assert d("a/b/foo bar.png") == "foo_bar"
    assert d("a/b/foo!.png") == "foo_"
    assert d("a/b/[bracketed].jpg") == "_bracketed_"
    # Empty / whitespace-only input -> None.
    assert d("") is None
    assert d("   ") is None
    # No-extension path (defensive — Pico-Banana always has extensions).
    # `Path("foo").stem == "foo"`, so this should produce a valid stem.
    assert d("foo") == "foo"


def test_no_id_collision_between_numeric_and_string_filenames():
    """Regression test for the actual production bug: ``1.png`` and
    ``kewsee_retry1.png`` must produce DIFFERENT triplet_ids.

    The old integer-extracting parser took the trailing digit from each,
    yielding ``picobanana_00000001`` for both. On the full Pico-Banana
    release this caused ~22% of triplet_ids to collide.
    """
    d = PicoBananaAdapter._derive_id_stem
    a = d("images/positive-edit/1.png")
    b = d("images/positive-edit/kewsee_retry1.png")
    assert a != b
    assert a == "1"
    assert b == "kewsee_retry1"
