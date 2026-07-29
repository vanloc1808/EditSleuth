#!/usr/bin/env python3
"""Generate a resumable free-form VLM chain baseline on a matched sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


def difficulty_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    bins = ("easy", "medium", "hard")
    base, remainder = divmod(n, len(bins))
    groups = []
    for index, bin_name in enumerate(bins):
        count = base + (1 if index < remainder else 0)
        group = df[df["reasoning_difficulty_bin"] == bin_name]
        groups.append(group.sample(n=count, random_state=seed + index))
    return pd.concat(groups).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_resized(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.BILINEAR,
        )
    return image


def prompt_messages(
    real: Image.Image,
    edited: Image.Image,
    instruction: str,
) -> list[dict]:
    prompt = (
        "You are auditing an image edit. Compare the original image (first) "
        "with the edited image (second), using the edit instruction only as "
        "context. Produce exactly six numbered steps:\n"
        "1. Restate the intended edit.\n"
        "2. Describe where visible changes occur and their approximate extent.\n"
        "3. Describe structural or visual differences.\n"
        "4. Classify the edit type.\n"
        "5. Describe forensic artifacts actually visible in this image pair; "
        "do not insert generic artifacts unless they are visually supported.\n"
        "6. Assess whether detecting the edit is easy, medium, or hard and why.\n"
        "Do not invent numerical measurements. Be concise.\n\n"
        f'Edit instruction: "{instruction}"'
    )
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": real},
            {"type": "image", "image": edited},
            {"type": "text", "text": prompt},
        ],
    }]


def read_completed(progress: Path) -> set[str]:
    if not progress.exists():
        return set()
    completed = set()
    with progress.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(json.loads(line)["triplet_id"])
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-max-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.with_suffix(".progress.jsonl")
    sample_path = output.with_suffix(".sample.parquet")
    if sample_path.exists():
        selected = pd.read_parquet(sample_path)
    else:
        annotations = pd.read_parquet(args.annotations)
        selected = difficulty_sample(annotations, args.n, args.seed)
        selected.to_parquet(sample_path, index=False)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    root = args.image_root.expanduser().resolve()
    completed = read_completed(progress)
    with progress.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(selected.to_dict("records"), start=1):
            if record["triplet_id"] in completed:
                continue
            real = load_resized(root / record["real_path"], args.image_max_side)
            edited = load_resized(root / record["edited_path"], args.image_max_side)
            messages = prompt_messages(real, edited, record["instruction"])
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(
                text=[prompt],
                images=[[real, edited]],
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            generated_ids = generated[:, inputs["input_ids"].shape[1]:]
            chain = processor.batch_decode(
                generated_ids, skip_special_tokens=True,
            )[0]
            result = {
                "triplet_id": record["triplet_id"],
                "difficulty_bin": record["reasoning_difficulty_bin"],
                "category": record["category_category"],
                "real_path": record["real_path"],
                "edited_path": record["edited_path"],
                "instruction": record["instruction"],
                "computed_chain": record["reasoning_chain"],
                "freeform_chain": chain,
                "model": args.model,
                "max_new_tokens": args.max_new_tokens,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(selected)} {record['triplet_id']}")

    rows = []
    with progress.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    by_id = {row["triplet_id"]: row for row in rows}
    ordered = [by_id[triplet_id] for triplet_id in selected["triplet_id"]]
    pd.DataFrame(ordered).to_parquet(output, index=False)
    summary = {
        "n": len(ordered),
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "difficulty_counts": pd.Series(
            [row["difficulty_bin"] for row in ordered]
        ).value_counts().to_dict(),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote matched free-form baseline to {output}")


if __name__ == "__main__":
    main()
