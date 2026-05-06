"""Generate the category × difficulty cross-tab for paper §4.

Reads the Stage E reasoning parquet and computes within-category
percentages of difficulty bins. Outputs the LaTeX table rows needed
to fill in ``tab:cat-difficulty`` in section_4_characterization.tex.

Usage::

    uv run python scripts/category_difficulty_crosstab.py \\
        --reasoning-parquet artifacts/reasoning/pico_banana.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Pretty-print order — matches the order in section_4_characterization.tex.
CATEGORY_ORDER = [
    "object_addition", "object_removal", "object_replacement",
    "attribute_change", "style_transfer", "photometric",
    "scene_transformation", "background_change", "text_edit",
    "geometric", "human_centric",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning-parquet", type=Path, required=True)
    args = parser.parse_args()

    shards = sorted(args.reasoning_parquet.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {args.reasoning_parquet}")
    df = pd.concat(
        [pd.read_parquet(s, columns=["category", "difficulty_bin"]) for s in shards],
        ignore_index=True,
    )
    n = len(df)

    # Joint counts and within-category percentages.
    joint = pd.crosstab(df["category"], df["difficulty_bin"])
    pct = pd.crosstab(df["category"], df["difficulty_bin"], normalize="index") * 100

    # Make sure all bins are columns (some categories may be empty in some bins).
    for col in ("easy", "medium", "hard"):
        if col not in pct.columns:
            pct[col] = 0.0
    pct = pct[["easy", "medium", "hard"]]

    print(f"# Within-category difficulty distribution (n_total = {n})\n")
    print(f"# {'category':<22} | {'count':>8} | {'easy':>6} | {'medium':>6} | {'hard':>6}")
    print(f"# {'-'*22} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6}")

    for cat in CATEGORY_ORDER:
        if cat not in pct.index:
            print(f"# {cat:<22} | {'0':>8} | {'-':>6} | {'-':>6} | {'-':>6}")
            continue
        cnt = int(joint.loc[cat].sum())
        e = pct.loc[cat, "easy"]
        m = pct.loc[cat, "medium"]
        h = pct.loc[cat, "hard"]
        print(f"# {cat:<22} | {cnt:>8} | {e:>6.1f} | {m:>6.1f} | {h:>6.1f}")

    # LaTeX table rows ready to drop into section_4_characterization.tex
    print("\n# === LaTeX table rows for tab:cat-difficulty ===\n")
    for cat in CATEGORY_ORDER:
        if cat not in pct.index:
            continue
        e = pct.loc[cat, "easy"]
        m = pct.loc[cat, "medium"]
        h = pct.loc[cat, "hard"]
        latex_cat = cat.replace("_", "\\_")
        print(f"    \\texttt{{{latex_cat}}} & {e:.1f} & {m:.1f} & {h:.1f} \\\\")


if __name__ == "__main__":
    main()
