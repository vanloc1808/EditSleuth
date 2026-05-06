"""Inspect a pilot-evaluation per-row predictions parquet.

The ``pilot_evaluate.py`` script writes a per-row parquet alongside
its summary JSON, containing every triplet's generated text plus
the extracted fields. This inspector prints a few rows so we can
verify what the model is actually generating and whether the
extractor is parsing it correctly.

Usage::

    uv run python scripts/inspect_pilot_predictions.py \\
        --predictions-parquet artifacts/pilot/chain_eval.per_row.parquet \\
        --n 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-parquet", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5,
                        help="number of rows to inspect")
    parser.add_argument("--max-text-len", type=int, default=600,
                        help="truncate generated text to this many chars")
    args = parser.parse_args()

    if not args.predictions_parquet.exists():
        raise FileNotFoundError(f"no parquet at {args.predictions_parquet}")

    df = pd.read_parquet(args.predictions_parquet)
    print(f"# total rows: {len(df)}")
    print(f"# columns: {list(df.columns)}")
    print()

    # Aggregate field-extraction success rates.
    fields_present = {
        "category_extracted": int(df["pred_category"].notna().sum()),
        "spatial_extracted": int(df["pred_spatial"].notna().sum()),
        "bin_extracted": int(df["pred_bin"].notna().sum()),
    }
    n = len(df)
    print(f"# field-extraction success rates:")
    for k, v in fields_present.items():
        print(f"#   {k}: {v}/{n} ({100*v/n:.1f}%)")
    print()

    # Show a few rows in detail.
    for i in range(min(args.n, len(df))):
        row = df.iloc[i]
        print(f"--- row {i}: triplet_id={row['triplet_id']} ---")
        print(f"  true_category:  {row['true_category']}")
        print(f"  pred_category:  {row['pred_category']}")
        print(f"  true_spatial:   {row['true_spatial']}")
        print(f"  pred_spatial:   {row['pred_spatial']}")
        print(f"  true_bin:       {row['true_bin']}")
        print(f"  pred_bin:       {row['pred_bin']}")
        text = row["generated_text"]
        if len(text) > args.max_text_len:
            text = text[:args.max_text_len] + f"... (+{len(row['generated_text']) - args.max_text_len} chars)"
        print(f"  generated_text:\n    {text}")
        print()


if __name__ == "__main__":
    main()
