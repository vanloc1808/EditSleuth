#!/usr/bin/env python3
"""Generate input-gradient saliency maps for a label-only Qwen2-VL adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from edit2forensics.pilot_prompt import format_user_turn


def normalize_saliency(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(values, [5, 99])
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def split_patch_saliency(
    patch_scores: np.ndarray,
    image_grid_thw: np.ndarray,
) -> list[np.ndarray]:
    maps = []
    offset = 0
    for t, h, w in image_grid_thw.astype(int):
        count = int(t * h * w)
        chunk = patch_scores[offset:offset + count]
        if len(chunk) != count:
            raise ValueError("pixel-value saliency does not match image_grid_thw")
        maps.append(normalize_saliency(chunk.reshape(t, h, w).mean(axis=0)))
        offset += count
    if offset != len(patch_scores):
        raise ValueError("unused pixel-value saliency rows remain after grid split")
    return maps


def save_overlay(image: Image.Image, saliency: np.ndarray, prefix: Path) -> None:
    image = image.convert("RGB")
    heat = Image.fromarray(np.uint8(saliency * 255), mode="L").resize(
        image.size, Image.Resampling.BILINEAR,
    )
    alpha = np.asarray(heat, dtype=np.float32) / 255.0
    color = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    color[..., 0] = np.uint8(255 * alpha)
    color[..., 1] = np.uint8(180 * np.sqrt(alpha))
    base = np.asarray(image, dtype=np.float32)
    blend = np.clip(
        base * (1.0 - 0.55 * alpha[..., None])
        + color.astype(np.float32) * (0.55 * alpha[..., None]),
        0,
        255,
    ).astype(np.uint8)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    heat.save(prefix.with_name(prefix.name + "_heatmap.png"))
    Image.fromarray(blend).save(prefix.with_name(prefix.name + "_overlay.png"))


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    groups = []
    grouped = list(df.groupby(["reasoning_difficulty_bin", "category_category"]))
    base, remainder = divmod(n, len(grouped))
    for index, (_, group) in enumerate(grouped):
        take = min(base + (1 if index < remainder else 0), len(group))
        if take:
            groups.append(group.sample(n=take, random_state=seed + index))
    sampled = pd.concat(groups, ignore_index=True)
    if len(sampled) < n:
        remaining = df[~df["triplet_id"].isin(sampled["triplet_id"])]
        sampled = pd.concat([
            sampled,
            remaining.sample(n=n - len(sampled), random_state=seed),
        ])
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_image(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.BILINEAR,
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-max-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--include-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    annotations = pd.read_parquet(args.annotations)
    selected = stratified_sample(annotations, args.n, args.seed)
    root = args.image_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.adapter_path)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.device
    rows = []
    for index, record in enumerate(selected.to_dict("records"), start=1):
        triplet_id = record["triplet_id"]
        real = load_image(root / record["real_path"], args.image_max_side)
        edited = load_image(root / record["edited_path"], args.image_max_side)
        messages = format_user_turn(
            real, edited, record["instruction"], args.include_instruction,
        )
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt], images=[[real, edited]], return_tensors="pt", padding=True,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = generated[:, prompt_length:]
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0]

        full_ids = torch.cat([inputs["input_ids"], generated_ids], dim=1)
        full_attention = torch.cat([
            inputs["attention_mask"],
            torch.ones_like(generated_ids, device=device),
        ], dim=1)
        pixel_values = inputs["pixel_values"].detach().requires_grad_(True)
        forward_kwargs = {
            key: value
            for key, value in inputs.items()
            if key not in {"input_ids", "attention_mask", "pixel_values"}
        }
        model.zero_grad(set_to_none=True)
        result = model(
            input_ids=full_ids,
            attention_mask=full_attention,
            pixel_values=pixel_values,
            use_cache=False,
            **forward_kwargs,
        )
        token_logits = result.logits[:, prompt_length - 1:-1, :]
        objective = -functional.cross_entropy(
            token_logits.reshape(-1, token_logits.shape[-1]),
            generated_ids.reshape(-1),
            reduction="sum",
        )
        objective.backward()
        gradient = pixel_values.grad
        if gradient is None:
            raise RuntimeError("model did not produce pixel-value gradients")
        patch_scores = (
            gradient.detach().float().abs()
            * pixel_values.detach().float().abs()
        ).mean(dim=-1).cpu().numpy()
        grids = inputs["image_grid_thw"].detach().cpu().numpy()
        saliency_maps = split_patch_saliency(patch_scores, grids)
        if len(saliency_maps) != 2:
            raise RuntimeError(f"expected two image saliency maps, got {len(saliency_maps)}")
        sample_dir = output / triplet_id
        save_overlay(real, saliency_maps[0], sample_dir / "real")
        save_overlay(edited, saliency_maps[1], sample_dir / "edited")
        metadata = {
            "triplet_id": triplet_id,
            "difficulty_bin": record["reasoning_difficulty_bin"],
            "category": record["category_category"],
            "generated_text": generated_text,
            "objective_log_probability": float(objective.detach().cpu()),
            "include_instruction": args.include_instruction,
            "real_overlay": str((sample_dir / "real_overlay.png").relative_to(output)),
            "edited_overlay": str((sample_dir / "edited_overlay.png").relative_to(output)),
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append(metadata)
        print(f"{index}/{len(selected)} {triplet_id}")
    pd.DataFrame(rows).to_parquet(output / "saliency_index.parquet", index=False)
    print(f"Wrote {len(rows)} saliency examples to {output}")


if __name__ == "__main__":
    main()
