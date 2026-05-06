"""Quick diagnostic: scan triplets parquet for unique source_edit_type
values with counts. Used to populate the Pico-Banana label map after
discovering 94% of source labels are unmapped.

Usage::

    uv run python scripts/scan_source_labels.py \\
        --triplets-parquet artifacts/triplets/pico_banana.parquet \\
        [--output-csv artifacts/reports/source_labels_histogram.csv]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-parquet", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--metadata-key", default="source_edit_type",
                        help="Metadata key to extract (default: source_edit_type)")
    args = parser.parse_args()

    shards = sorted(args.triplets_parquet.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {args.triplets_parquet}")

    counts: Counter = Counter()
    n_total = 0
    n_missing = 0
    for shard in shards:
        df = pd.read_parquet(shard, columns=["metadata"])
        for raw_meta in df["metadata"]:
            n_total += 1
            if isinstance(raw_meta, str):
                try:
                    meta = json.loads(raw_meta)
                except json.JSONDecodeError:
                    n_missing += 1
                    continue
            elif isinstance(raw_meta, dict):
                meta = raw_meta
            else:
                n_missing += 1
                continue
            label = meta.get(args.metadata_key)
            if label is None or (isinstance(label, str) and not label.strip()):
                n_missing += 1
                continue
            counts[str(label).strip()] += 1

    # Print frequency-sorted histogram.
    print(f"# Source-label histogram for `{args.metadata_key}`")
    print(f"# n_total = {n_total}, n_missing = {n_missing}, "
          f"n_unique = {len(counts)}")
    print()
    print(f"{'count':>10}  {'pct':>6}  label")
    print("-" * 70)
    cum = 0
    for label, count in counts.most_common():
        cum += count
        pct = 100 * count / n_total
        print(f"{count:>10}  {pct:>5.2f}%  {label!r}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(
            counts.most_common(), columns=["label", "count"]
        )
        out_df.to_csv(args.output_csv, index=False)
        print(f"\nHistogram written to {args.output_csv}")


if __name__ == "__main__":
    main()
