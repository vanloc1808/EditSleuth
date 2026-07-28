#!/usr/bin/env python3
"""Create and summarize a difficulty-stratified human trace audit.

Workflow:

1. ``sample`` writes a 200-row CSV balanced across easy/medium/hard.
2. Annotators fill ``step_N_correct`` with yes/no/unclear and optionally
   ``step_N_error_type`` and ``step_N_notes``.
3. ``report`` emits per-step error rates, including breakdowns by difficulty
   and the first failing step for each trace.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

VALID_JUDGMENTS = {"yes", "no", "unclear"}


def split_six_steps(chain: str) -> list[str]:
    """Split the deterministic top-level 1..6 format.

    Boundaries require ``N. `` at the beginning of a line. This avoids
    treating common instruction-internal lists such as ``1.)`` as steps.
    """
    matches = list(re.finditer(r"(?m)^([1-6])\. ", chain or ""))
    if [int(match.group(1)) for match in matches] != list(range(1, 7)):
        raise ValueError("trace does not contain exactly ordered top-level steps 1..6")
    return [
        chain[match.start():matches[index + 1].start()].strip()
        if index + 1 < len(matches)
        else chain[match.start():].strip()
        for index, match in enumerate(matches)
    ]


def _allocation(total: int) -> dict[str, int]:
    bins = ("easy", "medium", "hard")
    base, remainder = divmod(total, len(bins))
    return {
        bin_name: base + (1 if index < remainder else 0)
        for index, bin_name in enumerate(bins)
    }


def sample(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)
    for column in (args.id_column, args.text_column, args.difficulty_column):
        if column not in df:
            raise ValueError(f"input is missing column {column!r}")

    groups = []
    for bin_name, count in _allocation(args.n).items():
        group = df[df[args.difficulty_column] == bin_name]
        if len(group) < count:
            raise ValueError(
                f"difficulty={bin_name} has {len(group)} rows; need {count}"
            )
        groups.append(group.sample(n=count, random_state=args.seed))
    selected = pd.concat(groups).sample(
        frac=1, random_state=args.seed,
    ).reset_index(drop=True)

    rows = []
    for record in selected.to_dict("records"):
        steps = split_six_steps(str(record[args.text_column]))
        row = {
            "triplet_id": record[args.id_column],
            "difficulty_bin": record[args.difficulty_column],
            "trace": record[args.text_column],
            "annotator": "",
            "overall_notes": "",
        }
        for index, text in enumerate(steps, start=1):
            row[f"step_{index}_text"] = text
            row[f"step_{index}_correct"] = ""
            row[f"step_{index}_error_type"] = ""
            row[f"step_{index}_notes"] = ""
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} traces to {output}")


def _normalize_judgment(value: object) -> str:
    judgment = str(value).strip().lower()
    if judgment not in VALID_JUDGMENTS:
        raise ValueError(
            f"invalid judgment {value!r}; expected yes, no, or unclear"
        )
    return judgment


def report(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input, keep_default_na=False)
    long_rows = []
    trace_rows = []
    for record in df.to_dict("records"):
        judgments = []
        for step in range(1, 7):
            judgment = _normalize_judgment(record[f"step_{step}_correct"])
            judgments.append(judgment)
            long_rows.append({
                "triplet_id": record["triplet_id"],
                "difficulty_bin": record["difficulty_bin"],
                "step": step,
                "judgment": judgment,
                "is_error": judgment == "no",
                "is_unclear": judgment == "unclear",
                "error_type": record.get(f"step_{step}_error_type", ""),
            })
        failing = [index for index, value in enumerate(judgments, 1) if value == "no"]
        trace_rows.append({
            "triplet_id": record["triplet_id"],
            "difficulty_bin": record["difficulty_bin"],
            "first_error_step": failing[0] if failing else None,
            "complete_chain_correct": all(value == "yes" for value in judgments),
            "has_unclear_step": "unclear" in judgments,
        })

    long_df = pd.DataFrame(long_rows)
    traces_df = pd.DataFrame(trace_rows)
    per_step = (
        long_df.groupby("step")
        .agg(
            n=("triplet_id", "size"),
            errors=("is_error", "sum"),
            unclear=("is_unclear", "sum"),
            error_rate=("is_error", "mean"),
        )
        .reset_index()
    )
    by_difficulty = (
        long_df.groupby(["difficulty_bin", "step"])
        .agg(
            n=("triplet_id", "size"),
            errors=("is_error", "sum"),
            unclear=("is_unclear", "sum"),
            error_rate=("is_error", "mean"),
        )
        .reset_index()
    )
    first_error = (
        traces_df["first_error_step"].value_counts(dropna=False).to_dict()
    )
    summary = {
        "n_traces": len(traces_df),
        "complete_chain_accuracy": float(traces_df["complete_chain_correct"].mean()),
        "traces_with_unclear_step": int(traces_df["has_unclear_step"].sum()),
        "first_error_step_counts": {
            ("none" if pd.isna(key) else str(int(key))): int(value)
            for key, value in first_error.items()
        },
        "per_step": per_step.to_dict("records"),
        "by_difficulty": by_difficulty.to_dict("records"),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    per_step.to_csv(output.with_suffix(".per_step.csv"), index=False)
    by_difficulty.to_csv(output.with_suffix(".by_difficulty.csv"), index=False)
    traces_df.to_csv(output.with_suffix(".per_trace.csv"), index=False)
    print(f"Wrote audit report to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--input", type=Path, required=True)
    sample_parser.add_argument("--output", type=Path, required=True)
    sample_parser.add_argument("--n", type=int, default=200)
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--id-column", default="triplet_id")
    sample_parser.add_argument("--text-column", default="reasoning_chain")
    sample_parser.add_argument(
        "--difficulty-column", default="reasoning_difficulty_bin",
    )
    sample_parser.set_defaults(func=sample)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--input", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
