"""Quick-inspect a parquet directory: print its schema, row count,
and a sample row.

Useful for diagnosing schema-mismatch errors during pipeline stages
where the error message points at a missing field (e.g.
``'category'``) but does not say which parquet is missing the field.

Usage::

    uv run python scripts/inspect_parquet.py \\
        --path artifacts/categories/magicbrush_dev.parquet

Output: column list, row count, dtypes, and one sample row's contents
(truncated to keep wide string fields readable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True,
                        help="parquet directory")
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument("--max-string-len", type=int, default=100)
    args = parser.parse_args()

    shards = sorted(args.path.glob("part-*.parquet"))
    if not shards:
        print(f"ERROR: no parquet shards under {args.path}")
        print(f"       directory exists: {args.path.exists()}")
        if args.path.exists():
            print(f"       directory contents:")
            for p in sorted(args.path.iterdir())[:20]:
                print(f"         {p.name}")
        return

    print(f"# parquet directory: {args.path}")
    print(f"# n shards: {len(shards)}")

    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    print(f"# n rows total: {len(df)}")
    print()
    print(f"# columns ({len(df.columns)}):")
    for col, dtype in df.dtypes.items():
        print(f"#   {col:<30} {dtype}")
    print()

    print(f"# sample rows (first {args.sample_rows}):")
    for i in range(min(args.sample_rows, len(df))):
        print(f"# --- row {i} ---")
        for col, val in df.iloc[i].items():
            if isinstance(val, str) and len(val) > args.max_string_len:
                val = val[:args.max_string_len] + "..."
            print(f"#   {col}: {val!r}")
        print()


if __name__ == "__main__":
    main()
