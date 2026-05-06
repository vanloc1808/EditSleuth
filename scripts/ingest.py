"""Ingest stage: run an adapter and serialize `EditTriplet`s to parquet.

Usage::

    uv run python scripts/ingest.py \\
        adapter=pico_banana \\
        paths.dataset_root=/path/to/pico-banana-400k

Smoke test with a record cap::

    uv run python scripts/ingest.py \\
        adapter=pico_banana \\
        paths.dataset_root=/path/to/pico-banana-400k \\
        adapter.max_records=100

The output parquet has one row per EditTriplet, with columns matching
`EditTriplet.to_dict()` plus a JSON-encoded `metadata` column.
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

log = logging.getLogger(__name__)


def _flush_batch(rows: list[dict], out_path: Path, first_write: bool) -> None:
    """Write a batch of rows to parquet, appending via row-group concatenation.

    PyArrow does not natively support streaming append to a single parquet
    file, so we write one file per batch and merge at the end. For simplicity
    and robustness on large datasets, we instead write numbered shard files
    and leave them as a dataset directory — readers treat the directory
    as a single logical table.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Metadata must be JSON-encoded for parquet; it's a free-form dict.
    df["metadata"] = df["metadata"].apply(json.dumps)
    shard_idx = 0 if first_write else len(list(out_path.glob("part-*.parquet")))
    shard_path = out_path / f"part-{shard_idx:05d}.parquet"
    df.to_parquet(shard_path, index=False)
    log.info("wrote %d rows to %s", len(df), shard_path.name)


@hydra.main(version_base=None, config_path="../configs", config_name="ingest")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # --- Instantiate adapter ------------------------------------------------
    # We pop non-constructor fields (`source_dataset_tag`) before passing to
    # `instantiate`. Keeping them in the config lets us interpolate into
    # output paths without polluting the adapter's __init__ signature.
    adapter_cfg = OmegaConf.to_container(cfg.adapter, resolve=True)
    assert isinstance(adapter_cfg, dict)
    adapter_cfg.pop("source_dataset_tag", None)

    adapter = instantiate(adapter_cfg)
    log.info("Instantiated adapter: %s", type(adapter).__name__)

    # --- Prepare output directory (dataset of shards) -----------------------
    out_path = Path(cfg.output_path)
    # Treat the configured path as the dataset directory, whether or not it
    # ends in `.parquet` — this makes the path templating in configs clean.
    out_path.mkdir(parents=True, exist_ok=True)
    existing_shards = list(out_path.glob("part-*.parquet"))
    if existing_shards:
        log.warning(
            "output dir %s already contains %d shard(s); new shards will be "
            "appended with higher indices",
            out_path,
            len(existing_shards),
        )

    # --- Stream triplets through in batches --------------------------------
    batch: list[dict] = []
    total = 0
    first_write = not existing_shards

    for triplet in tqdm(adapter.ingest(), desc="ingest", unit="triplet"):
        batch.append(triplet.to_dict())
        total += 1
        if len(batch) >= cfg.write_batch_size:
            _flush_batch(batch, out_path, first_write=first_write)
            first_write = False
            batch.clear()

    _flush_batch(batch, out_path, first_write=first_write)
    log.info("ingestion complete: %d triplets written to %s", total, out_path)


if __name__ == "__main__":
    main()
