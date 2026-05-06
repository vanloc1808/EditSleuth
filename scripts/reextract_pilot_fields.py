"""Re-extract structured fields from a pilot-evaluation predictions parquet.

The original ``pilot_evaluate.py`` runs inference (~30-45 min) and
writes both a summary JSON and a per-row predictions parquet. The
parquet preserves every triplet's full ``generated_text``, so we
can re-run *only* the field-extraction step against an updated
extractor without paying the inference cost again.

This is the right tool when:
* A bug was found in the extractor regex.
* The extractor was tightened to handle additional prose patterns.
* You want to change which fields count as "matched" (e.g., relaxing
  category matching to accept space-separated forms).

Usage::

    uv run python scripts/reextract_pilot_fields.py \\
        --predictions-parquet artifacts/pilot/chain_eval.per_row.parquet \\
        --target-mode chain \\
        --output-summary artifacts/pilot/chain_eval_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Reuse the extractors from pilot_evaluate.py to keep the logic
# in one place.
sys.path.insert(0, str(Path(__file__).parent))
from pilot_evaluate import (  # noqa: E402
    _extract_chain_fields,
    _extract_label_only_fields,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-parquet", type=Path, required=True)
    parser.add_argument("--target-mode", choices=("chain", "label_only"),
                        required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-per-row", type=Path, default=None,
                        help="optional: write the updated per-row parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.predictions_parquet)
    extract = (_extract_chain_fields if args.target_mode == "chain"
               else _extract_label_only_fields)

    # Re-run extraction on every row's generated_text.
    new_predictions = df["generated_text"].apply(extract)
    df = df.copy()
    df["pred_category"] = new_predictions.apply(lambda d: d["category"])
    df["pred_spatial"] = new_predictions.apply(lambda d: d["spatial_descriptor"])
    df["pred_bin"] = new_predictions.apply(lambda d: d["difficulty_bin"])
    df["category_match"] = df["pred_category"] == df["true_category"]
    df["spatial_match"] = df["pred_spatial"] == df["true_spatial"]
    df["bin_match"] = df["pred_bin"] == df["true_bin"]

    summary = {
        "n": int(len(df)),
        "target_mode": args.target_mode,
        "category_accuracy": float(df["category_match"].mean()),
        "spatial_descriptor_accuracy": float(df["spatial_match"].mean()),
        "difficulty_bin_accuracy": float(df["bin_match"].mean()),
        "joint_field_accuracy": float(
            (df["category_match"]
             & df["spatial_match"]
             & df["bin_match"]).mean()
        ),
        # Field-extraction success rates: how often did the extractor
        # find each field (independent of correctness)? Useful for
        # diagnosing extractor coverage vs model accuracy.
        "category_extracted_rate": float(df["pred_category"].notna().mean()),
        "spatial_extracted_rate": float(df["pred_spatial"].notna().mean()),
        "bin_extracted_rate": float(df["pred_bin"].notna().mean()),
    }
    print("\n===== re-extracted summary =====")
    print(json.dumps(summary, indent=2))

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary to {args.output_summary}")

    if args.output_per_row is not None:
        args.output_per_row.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output_per_row, index=False)
        print(f"wrote per-row predictions to {args.output_per_row}")


if __name__ == "__main__":
    main()
