#!/usr/bin/env python3
"""Prepare and report a blinded paired computed-vs-free-form chain audit."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

JUDGMENTS = {"yes", "no", "unclear"}


def extract_steps(text: str) -> tuple[list[str], bool]:
    matches = list(re.finditer(r"(?m)^\s*(?:\*\*)?([1-6])\.\s+", text or ""))
    numbers = [int(match.group(1)) for match in matches]
    valid = numbers == list(range(1, 7))
    steps = [""] * 6
    for index, match in enumerate(matches):
        number = int(match.group(1))
        if 1 <= number <= 6 and not steps[number - 1]:
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            steps[number - 1] = text[match.start():end].strip()
    return steps, valid


def prepare(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)
    required = {
        "triplet_id", "difficulty_bin", "real_path", "edited_path",
        "instruction", "computed_chain", "freeform_chain",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"baseline output missing: {sorted(missing)}")
    bundle = args.bundle_dir.expanduser().resolve()
    images = bundle / "images"
    images.mkdir(parents=True, exist_ok=True)
    root = args.image_root.expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    annotation_rows = []
    key_rows = []
    for index, record in enumerate(df.to_dict("records"), start=1):
        audit_id = f"C{index:03d}"
        real_source = root / record["real_path"]
        edited_source = root / record["edited_path"]
        real_name = f"{audit_id}__real{real_source.suffix.lower()}"
        edited_name = f"{audit_id}__edited{edited_source.suffix.lower()}"
        shutil.copy2(real_source, images / real_name)
        shutil.copy2(edited_source, images / edited_name)
        computed_steps, computed_valid = extract_steps(record["computed_chain"])
        freeform_steps, freeform_valid = extract_steps(record["freeform_chain"])
        if rng.integers(0, 2) == 0:
            side_to_system = {"A": "computed", "B": "freeform"}
        else:
            side_to_system = {"A": "freeform", "B": "computed"}
        chains = {
            "computed": (record["computed_chain"], computed_steps, computed_valid),
            "freeform": (record["freeform_chain"], freeform_steps, freeform_valid),
        }
        key_rows.append({
            "audit_id": audit_id,
            "triplet_id": record["triplet_id"],
            "difficulty_bin": record["difficulty_bin"],
            "system_A": side_to_system["A"],
            "system_B": side_to_system["B"],
            "computed_format_valid": computed_valid,
            "freeform_format_valid": freeform_valid,
        })
        for annotator in range(1, args.annotators + 1):
            row = {
                "audit_id": audit_id,
                "annotator": f"A{annotator}",
                "real_image": f"images/{real_name}",
                "edited_image": f"images/{edited_name}",
                "instruction": record["instruction"],
                "overall_notes": "",
            }
            for side in ("A", "B"):
                system = side_to_system[side]
                raw, steps, _ = chains[system]
                row[f"trace_{side}"] = raw
                for step in range(1, 7):
                    row[f"{side}_step_{step}_text"] = steps[step - 1]
                    row[f"{side}_step_{step}_correct"] = ""
                    row[f"{side}_step_{step}_error_type"] = ""
                    row[f"{side}_step_{step}_notes"] = ""
            annotation_rows.append(row)
    pd.DataFrame(annotation_rows).to_csv(bundle / "annotations.csv", index=False)
    pd.DataFrame(key_rows).to_csv(bundle / "answer_key.csv", index=False)
    rubric = (
        "# Paired chain faithfulness audit\n\n"
        "Judge Trace A and Trace B independently against the image pair and "
        "instruction. For each step enter `yes`, `no`, or `unclear` in the "
        "correctness field. A statement is correct only if its instance-level "
        "claims are visually supported or accurately scoped. Mark generic "
        "priors as incorrect when presented as evidence actually visible in "
        "the current pair. Use consistent error types: `localization`, "
        "`magnitude`, `category`, `unsupported_prior`, `difficulty`, "
        "`hallucination`, or `format`. Annotators work independently and do "
        "not open `answer_key.csv` until completion.\n"
    )
    (bundle / "RUBRIC.md").write_text(rubric)
    print(f"Wrote paired audit bundle with {len(df)} triplets to {bundle}")


def _consensus(values: pd.Series) -> str:
    normalized = values.str.strip().str.lower()
    invalid = set(normalized) - JUDGMENTS
    if invalid:
        raise ValueError(f"invalid judgments: {sorted(invalid)}")
    numeric = normalized.map({"yes": 1.0, "no": 0.0, "unclear": np.nan})
    if numeric.notna().sum() == 0:
        return "unclear"
    mean = numeric.mean()
    if mean > 0.5:
        return "yes"
    if mean < 0.5:
        return "no"
    return "unclear"


def report(args: argparse.Namespace) -> None:
    annotations = pd.read_csv(args.annotations, keep_default_na=False)
    key = pd.read_csv(args.answer_key)
    records = []
    for row in annotations.to_dict("records"):
        key_row = key[key["audit_id"] == row["audit_id"]].iloc[0]
        for side in ("A", "B"):
            system = key_row[f"system_{side}"]
            for step in range(1, 7):
                judgment = str(row[f"{side}_step_{step}_correct"]).strip().lower()
                if judgment not in JUDGMENTS:
                    raise ValueError(
                        f"{row['audit_id']} {side} step {step}: {judgment!r}"
                    )
                records.append({
                    "audit_id": row["audit_id"],
                    "annotator": row["annotator"],
                    "difficulty_bin": key_row["difficulty_bin"],
                    "system": system,
                    "step": step,
                    "judgment": judgment,
                    "error_type": row[f"{side}_step_{step}_error_type"],
                })
    long = pd.DataFrame(records)
    consensus = (
        long.groupby(["audit_id", "difficulty_bin", "system", "step"])
        .agg(
            judgment=("judgment", _consensus),
            error_type=("error_type", lambda values: "|".join(
                sorted({value for value in values if str(value).strip()})
            )),
        )
        .reset_index()
    )
    consensus["is_error"] = consensus["judgment"] == "no"
    consensus["is_unclear"] = consensus["judgment"] == "unclear"
    per_step = (
        consensus.groupby(["system", "step"])
        .agg(
            n=("audit_id", "size"),
            errors=("is_error", "sum"),
            unclear=("is_unclear", "sum"),
            error_rate=("is_error", "mean"),
        )
        .reset_index()
    )
    by_difficulty = (
        consensus.groupby(["system", "difficulty_bin", "step"])
        .agg(
            n=("audit_id", "size"),
            errors=("is_error", "sum"),
            unclear=("is_unclear", "sum"),
            error_rate=("is_error", "mean"),
        )
        .reset_index()
    )
    trace_rows = []
    for (audit_id, system), group in consensus.groupby(["audit_id", "system"]):
        failures = group.loc[group["is_error"], "step"].tolist()
        trace_rows.append({
            "audit_id": audit_id,
            "system": system,
            "first_error_step": min(failures) if failures else None,
            "complete_chain_correct": bool((group["judgment"] == "yes").all()),
            "has_unclear": bool(group["is_unclear"].any()),
        })
    traces = pd.DataFrame(trace_rows)
    paired_tests = {}
    for step in range(1, 7):
        subset = consensus[consensus["step"] == step].pivot(
            index="audit_id", columns="system", values="judgment",
        )
        subset = subset[
            subset["computed"].isin({"yes", "no"})
            & subset["freeform"].isin({"yes", "no"})
        ]
        computed = subset["computed"] == "yes"
        freeform = subset["freeform"] == "yes"
        computed_only = int((computed & ~freeform).sum())
        freeform_only = int((~computed & freeform).sum())
        discordant = computed_only + freeform_only
        paired_tests[str(step)] = {
            "n_clear_pairs": len(subset),
            "computed_only_correct": computed_only,
            "freeform_only_correct": freeform_only,
            "mcnemar_exact_p": (
                float(binomtest(min(computed_only, freeform_only), discordant, 0.5).pvalue)
                if discordant else 1.0
            ),
        }
    summary = {
        "n_triplets": int(key["audit_id"].nunique()),
        "n_annotators": int(annotations["annotator"].nunique()),
        "format_valid": {
            "computed": float(key["computed_format_valid"].mean()),
            "freeform": float(key["freeform_format_valid"].mean()),
        },
        "complete_chain_accuracy": traces.groupby("system")[
            "complete_chain_correct"
        ].mean().to_dict(),
        "per_step": per_step.to_dict("records"),
        "by_difficulty": by_difficulty.to_dict("records"),
        "paired_step_tests": paired_tests,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    consensus.to_csv(output.with_suffix(".consensus_steps.csv"), index=False)
    traces.to_csv(output.with_suffix(".per_trace.csv"), index=False)
    print(f"Wrote paired chain audit report to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--image-root", type=Path, required=True)
    prepare_parser.add_argument("--bundle-dir", type=Path, required=True)
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
