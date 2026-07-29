#!/usr/bin/env python3
"""Prepare and summarize a blinded 100-triplet human difficulty audit."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LEVELS = ("easy", "medium", "hard")
LEVEL_TO_NUMBER = {level: index + 1 for index, level in enumerate(LEVELS)}


def balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Balance across difficulty/category cells, then fill deterministically."""
    required = {"reasoning_difficulty_bin", "category_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"release annotations missing: {sorted(missing)}")
    groups = list(df.groupby(["reasoning_difficulty_bin", "category_category"]))
    base = n // len(groups)
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    remaining_by_group: dict[tuple[str, str], list[int]] = {}
    for key, group in groups:
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        take = min(base, len(indices))
        selected_indices.extend(indices[:take].tolist())
        remaining_by_group[key] = indices[take:].tolist()

    categories = sorted({key[1] for key in remaining_by_group})
    keys = []
    for round_index in range(len(LEVELS)):
        for category_index, category in enumerate(categories):
            bin_name = LEVELS[(category_index + round_index) % len(LEVELS)]
            key = (bin_name, category)
            if key in remaining_by_group:
                keys.append(key)
    cursor = 0
    while len(selected_indices) < n:
        if not any(remaining_by_group.values()):
            raise ValueError(f"only {len(selected_indices)} rows available; need {n}")
        key = keys[cursor % len(keys)]
        if remaining_by_group[key]:
            selected_indices.append(remaining_by_group[key].pop())
        cursor += 1
    return df.loc[selected_indices].sample(
        frac=1, random_state=seed,
    ).reset_index(drop=True)


def prepare(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)
    required = {
        "triplet_id", "real_path", "edited_path", "instruction",
        "reasoning_difficulty_bin", "difficulty_difficulty_raw",
        "category_category",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"release annotations missing: {sorted(missing)}")
    selected = balanced_sample(df, args.n, args.seed)

    bundle = args.bundle_dir.expanduser().resolve()
    images = bundle / "images"
    images.mkdir(parents=True, exist_ok=True)
    root = args.image_root.expanduser().resolve()
    key_rows = []
    annotation_rows = []
    for audit_index, record in enumerate(selected.to_dict("records"), start=1):
        audit_id = f"D{audit_index:03d}"
        real_source = root / record["real_path"]
        edited_source = root / record["edited_path"]
        if not real_source.exists() or not edited_source.exists():
            raise FileNotFoundError(
                f"missing image pair for {record['triplet_id']}: "
                f"{real_source}, {edited_source}"
            )
        real_name = f"{audit_id}__real{real_source.suffix.lower()}"
        edited_name = f"{audit_id}__edited{edited_source.suffix.lower()}"
        shutil.copy2(real_source, images / real_name)
        shutil.copy2(edited_source, images / edited_name)
        key_rows.append({
            "audit_id": audit_id,
            "triplet_id": record["triplet_id"],
            "computed_difficulty_bin": record["reasoning_difficulty_bin"],
            "computed_difficulty_raw": record["difficulty_difficulty_raw"],
            "category": record["category_category"],
        })
        for annotator in range(1, args.annotators + 1):
            annotation_rows.append({
                "audit_id": audit_id,
                "annotator": f"A{annotator}",
                "real_image": f"images/{real_name}",
                "edited_image": f"images/{edited_name}",
                "instruction": record["instruction"],
                "human_difficulty": "",
                "confidence_1_to_5": "",
                "notes": "",
            })

    pd.DataFrame(annotation_rows).to_csv(bundle / "annotations.csv", index=False)
    pd.DataFrame(key_rows).to_csv(bundle / "answer_key.csv", index=False)
    rubric = (
        "# Human difficulty audit rubric\n\n"
        "Judge how difficult it is to identify and characterize the edit from "
        "the real/edited image pair. Use `easy`, `medium`, or `hard`.\n\n"
        "- easy: edit location/type is immediately clear\n"
        "- medium: requires careful comparison but remains identifiable\n"
        "- hard: subtle, diffuse, ambiguous, or difficult to characterize\n\n"
        "Do not open `answer_key.csv` until annotation is complete. Record a "
        "confidence from 1 (low) to 5 (high). Annotators work independently.\n"
    )
    (bundle / "RUBRIC.md").write_text(rubric)
    print(f"Wrote blinded {args.n}-triplet audit bundle to {bundle}")


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    n_levels = len(LEVELS)
    observed = np.zeros((n_levels, n_levels), dtype=float)
    for left, right in zip(a, b):
        observed[left - 1, right - 1] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    expected /= observed.sum()
    weights = np.fromfunction(
        lambda i, j: ((i - j) / (n_levels - 1)) ** 2,
        (n_levels, n_levels),
    )
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator else 1.0


def report(args: argparse.Namespace) -> None:
    annotations = pd.read_csv(args.annotations, keep_default_na=False)
    key = pd.read_csv(args.answer_key)
    normalized = annotations["human_difficulty"].str.strip().str.lower()
    invalid = sorted(set(normalized) - set(LEVELS))
    if invalid:
        raise ValueError(f"invalid human_difficulty values: {invalid}")
    annotations = annotations.copy()
    annotations["human_difficulty"] = normalized
    annotations["human_number"] = normalized.map(LEVEL_TO_NUMBER)
    joined = annotations.merge(key, on="audit_id", validate="many_to_one")
    joined["computed_number"] = joined["computed_difficulty_bin"].map(
        LEVEL_TO_NUMBER,
    )

    per_annotator = {}
    for annotator, group in joined.groupby("annotator"):
        rho, p = spearmanr(
            group["computed_difficulty_raw"], group["human_number"],
        )
        per_annotator[annotator] = {
            "n": len(group),
            "bin_accuracy": float(
                (group["human_difficulty"] == group["computed_difficulty_bin"]).mean()
            ),
            "spearman_raw_vs_human": float(rho),
            "spearman_p": float(p),
        }

    pivot = joined.pivot(
        index="audit_id", columns="annotator", values="human_number",
    )
    agreement = {}
    if pivot.shape[1] == 2:
        left = pivot.iloc[:, 0].to_numpy(dtype=int)
        right = pivot.iloc[:, 1].to_numpy(dtype=int)
        agreement = {
            "annotators": list(pivot.columns),
            "exact_agreement": float((left == right).mean()),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(left, right),
        }

    consensus = (
        joined.groupby("audit_id")["human_number"].median().round().astype(int)
        .rename("consensus_number")
    )
    consensus_df = key.merge(consensus, on="audit_id", validate="one_to_one")
    consensus_df["consensus_difficulty"] = consensus_df["consensus_number"].map(
        {value: key_ for key_, value in LEVEL_TO_NUMBER.items()},
    )
    rho, p = spearmanr(
        consensus_df["computed_difficulty_raw"],
        consensus_df["consensus_number"],
    )
    confusion = pd.crosstab(
        consensus_df["computed_difficulty_bin"],
        consensus_df["consensus_difficulty"],
    ).reindex(index=LEVELS, columns=LEVELS, fill_value=0)
    summary = {
        "n_triplets": len(consensus_df),
        "n_annotations": len(joined),
        "per_annotator": per_annotator,
        "inter_annotator": agreement,
        "consensus": {
            "bin_accuracy": float(
                (
                    consensus_df["computed_difficulty_bin"]
                    == consensus_df["consensus_difficulty"]
                ).mean()
            ),
            "spearman_raw_vs_human": float(rho),
            "spearman_p": float(p),
            "confusion": confusion.to_dict(),
        },
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    consensus_df.to_csv(output.with_suffix(".per_triplet.csv"), index=False)
    print(f"Wrote human difficulty report to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--image-root", type=Path, required=True)
    prepare_parser.add_argument("--bundle-dir", type=Path, required=True)
    prepare_parser.add_argument("--n", type=int, default=100)
    prepare_parser.add_argument("--annotators", type=int, default=2)
    prepare_parser.add_argument("--seed", type=int, default=2026)
    prepare_parser.set_defaults(func=prepare)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--annotations", type=Path, required=True)
    report_parser.add_argument("--answer-key", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
