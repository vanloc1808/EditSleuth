"""Stage B entry point: generate auto-masks for ingested triplets.

Consumes a parquet directory produced by `scripts/ingest.py` and emits:

1. Per-triplet mask PNGs under ``masks_root/<triplet_id>.png``.
2. A parquet dataset of `MaskArtifact`s at ``output_path``. Each output
   shard mirrors its corresponding input shard's filename, so output
   sharding follows input sharding 1:1.

Usage::

    uv run python scripts/generate_masks.py \\
        triplets_path=/path/to/artifacts/triplets/pico_banana.parquet \\
        paths.output_root=/path/to/artifacts

Hydra composes `mask_generator` (signal choice) and `paths` configs.

Resumability and parallelism
----------------------------
The driver is resumable and shard-parallelizable by design.

* **Resumable**: if an output shard's parquet already exists, the
  driver skips it (unless ``overwrite_existing_shards=true``). After
  a partial run is interrupted, simply re-invoke the driver — it will
  resume from where it left off.

* **Process-level parallelism via shard slicing**: the
  ``shard_indices`` config knob restricts processing to a chosen
  subset of input shards. Different processes can handle disjoint
  index lists concurrently without filename collisions, since each
  output mirrors its input shard's name. Example for an 8-way split
  using xargs::

      # Discover input shard count
      N=$(ls artifacts/triplets/pico_banana.parquet/part-*.parquet | wc -l)

      # Launch 8 workers, each handling 1/8 of the shards
      seq 0 $((N-1)) | xargs -n1 -P8 -I{} \\
          uv run python scripts/generate_masks.py \\
              triplets_path=artifacts/triplets/pico_banana.parquet \\
              paths.output_root=artifacts \\
              run_tag=pico_banana \\
              "shard_indices=[{}]"

  Workers don't communicate; output correctness is guaranteed by the
  filename-mirroring rule.

* **GPU note**: when LPIPS or DINOv2 is on the GPU, do *not* run
  multiple parallel workers — they will contend for GPU memory and
  likely run slower than a single sequential worker. Parallelism is
  for CPU-bound configurations (LAB-only or LPIPS-on-CPU).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from edit2forensics.data.triplet import EditTriplet

log = logging.getLogger(__name__)


def _flush_batch(rows: list[dict], out_dir: Path, shard_name: str) -> None:
    """Write a batch of artifact dicts to a parquet shard.

    The output filename is derived from the *input* shard name passed in,
    not from a counter on existing files. This makes parallel workers
    safe — two concurrent workers writing different input shards never
    pick the same output filename, even though they may be writing to
    the same directory.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_path = out_dir / shard_name
    df.to_parquet(out_path, index=False)
    log.info("wrote %d mask artifacts to %s", len(df), shard_name)


def _select_shards(triplets_path: Path, shard_indices) -> list[Path]:
    """Resolve which shards to process.

    Parameters
    ----------
    triplets_path
        Directory containing the triplets parquet shards.
    shard_indices
        Either ``None`` (process all shards) or a list of integer
        indices into the sorted-shard list. Indices must be in range;
        out-of-range indices raise immediately so a typo in a parallel
        launcher fails fast rather than silently doing the wrong thing.
    """
    all_shards = sorted(triplets_path.glob("part-*.parquet"))
    if not all_shards:
        raise FileNotFoundError(
            f"no parquet shards under {triplets_path}; did you run ingest.py?"
        )
    if shard_indices is None:
        return all_shards

    indices = list(shard_indices)
    n = len(all_shards)
    bad = [i for i in indices if i < 0 or i >= n]
    if bad:
        raise IndexError(
            f"shard_indices={indices} contains out-of-range entries {bad} "
            f"(only {n} shards available, indices 0..{n-1})"
        )
    return [all_shards[i] for i in indices]


def _iter_shard_triplets(shard: Path):
    """Yield (shard_name, EditTriplet) for one specific shard."""
    df = pd.read_parquet(shard)
    for _, row in df.iterrows():
        d = row.to_dict()
        # metadata was JSON-encoded during ingest; decode for audit use.
        if isinstance(d.get("metadata"), str):
            d["metadata"] = json.loads(d["metadata"])
        yield EditTriplet.from_dict(d)


@hydra.main(version_base=None, config_path="../configs", config_name="generate_masks")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # --- Instantiate the generator (signals + config + class in one call) ---
    generator = instantiate(cfg.mask_generator)
    log.info(
        "Signals: %s", [type(s).__name__ for s in generator.signals]
    )

    # --- Prepare outputs ---------------------------------------------------
    triplets_path = Path(cfg.triplets_path)
    masks_root = Path(cfg.masks_root)
    masks_root.mkdir(parents=True, exist_ok=True)

    out_dir = Path(cfg.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_indices = cfg.get("shard_indices")
    if shard_indices is not None:
        # Hydra/OmegaConf may give us a ListConfig; coerce to list[int].
        shard_indices = [int(i) for i in shard_indices]
    selected = _select_shards(triplets_path, shard_indices)
    log.info(
        "processing %d/%d input shards%s",
        len(selected),
        len(sorted(triplets_path.glob("part-*.parquet"))),
        f" (indices={shard_indices})" if shard_indices is not None else "",
    )

    # --- Stream, one input shard at a time --------------------------------
    # Output filenames mirror input shard names exactly. This lets us:
    #   1. Skip shards whose output already exists (resumable runs).
    #   2. Run multiple workers in parallel without filename collisions
    #      (each worker handles a disjoint subset of shard_indices).
    total = 0
    failed = 0
    skipped_done = 0

    for shard in selected:
        out_path = out_dir / shard.name
        if out_path.exists() and not cfg.get("overwrite_existing_shards", False):
            log.info("output shard %s already exists; skipping", shard.name)
            skipped_done += 1
            continue

        # Determine the driver-side batch size. We default to the
        # max LPIPS batch size if any LPIPS signal is in the stack
        # (since LPIPS is the dominant cost), else 1 (no batching needed
        # for pure-CPU stacks).
        from edit2forensics.mask.signals import LPIPSDiff
        lpips_signals = [s for s in generator.signals if isinstance(s, LPIPSDiff)]
        if lpips_signals:
            driver_batch_size = max(s.batch_size for s in lpips_signals)
        else:
            driver_batch_size = int(cfg.get("driver_batch_size", 1))
        log.info("driver batch size: %d", driver_batch_size)

        batch_artifacts: list[dict] = []
        shard_total = 0
        shard_failed = 0
        # Pre-buffer triplets up to driver_batch_size, then call
        # generate_batch, then flush results.
        triplet_buf: list[EditTriplet] = []
        path_buf: list[Path] = []
        progress = tqdm(
            _iter_shard_triplets(shard),
            desc=f"mask-gen[{shard.name}]",
            unit="triplet",
        )

        def _flush_triplet_buffer():
            """Run generate_batch on the buffered triplets and append
            their artifacts to the shard's output batch."""
            nonlocal shard_total, shard_failed
            if not triplet_buf:
                return
            try:
                arts = generator.generate_batch(triplet_buf, path_buf)
            except Exception as e:
                # Whole-batch failure (rare — usually a config or model
                # issue rather than per-triplet data). Fall back to
                # per-triplet so we don't lose the whole batch's worth
                # of work to one bad row.
                log.warning(
                    "batch of %d failed (%s); falling back to per-triplet",
                    len(triplet_buf), e,
                )
                arts = []
                for t, p in zip(triplet_buf, path_buf):
                    try:
                        arts.append(generator.generate(t, p))
                    except Exception as e2:
                        log.warning(
                            "mask generation failed for %s: %s", t.triplet_id, e2
                        )
                        shard_failed += 1
            for a in arts:
                batch_artifacts.append(a.to_dict())
                shard_total += 1
            triplet_buf.clear()
            path_buf.clear()

        for triplet in progress:
            mask_path = masks_root / f"{triplet.triplet_id}.png"
            triplet_buf.append(triplet)
            path_buf.append(mask_path)
            if len(triplet_buf) >= driver_batch_size:
                _flush_triplet_buffer()

        _flush_triplet_buffer()  # final partial batch

        # Flush all of the shard's results into one output parquet.
        # We deliberately use a single output per input shard rather
        # than honoring write_batch_size as a mid-shard flush boundary —
        # mid-shard flushing was useful only when output names were
        # counter-derived; with shard-mirrored names it complicates
        # parallel correctness for no win.
        _flush_batch(batch_artifacts, out_dir, shard_name=shard.name)
        total += shard_total
        failed += shard_failed

    log.info(
        "mask-gen complete: %d artifacts written to %s (%d failed, %d shards skipped as already-done)",
        total, out_dir, failed, skipped_done,
    )


if __name__ == "__main__":
    main()
