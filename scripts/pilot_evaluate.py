"""Pilot evaluation script: measure category accuracy, spatial-descriptor
accuracy, and chain faithfulness on a held-out evaluation set.

Loads a trained LoRA adapter from ``scripts/pilot_train.py`` and runs
inference over the held-out reasoning artifacts. Extracts structured
fields from the model's generated text and compares them against the
ground-truth artifacts. Reports per-metric accuracies and (for the
chain-target variant) an overall chain-faithfulness rate.

Pilot scope: greedy decoding, single GPU, full evaluation in one pass.
For benchmarking against multiple checkpoints or with sampling, extend
the inference loop accordingly.

Usage::

    uv run python scripts/pilot_evaluate.py \\
        annotations_parquet=editsleuth_data/release/magicbrush_dev_annotations.parquet \\
        image_root=/data/magicbrush \\
        adapter_path=artifacts/pilot/chain_run \\
        target_mode=chain \\
        output_path=artifacts/pilot/chain_eval.json
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from edit2forensics.pilot_prompt import format_user_turn

log = logging.getLogger(__name__)

# Canonical category strings the model is expected to produce; used
# to extract category from generated text.
CANONICAL_CATEGORIES = (
    "object_addition", "object_removal", "object_replacement",
    "attribute_change", "style_transfer", "photometric",
    "scene_transformation", "background_change", "text_edit",
    "geometric", "human_centric", "other",
)
CANONICAL_SPATIAL = (
    "whole_image", "upper_left", "upper_right",
    "lower_left", "lower_right", "centered", "scattered",
    "alignment_failed",
)


def _extract_chain_fields(generated_text: str) -> dict:
    """Parse a chain-target generation for category, spatial descriptor,
    and difficulty bin. Returns a dict; missing fields map to None.

    The chain format from Stage~E is six numbered prose steps. Stage~E
    also emits a structured one-line header ``[category=..., scope=...,
    difficulty=...]``, but the curriculum dataset feeds only the chain
    prose (not the header) to the model as the supervised target. So
    the model never learns to emit the header, and the extractor must
    read everything from prose.

    Extraction strategy per field:

    * **category**: prose step 4 says ``"4. The edit is classified as
      <category>, ..."``. We accept either the canonical underscore form
      (``object_addition``) or the natural-language space form
      (``object addition``) since the model may drift between them.
    * **spatial descriptor**: prose step 2 includes one of seven
      distinctive phrasings (``spans the entire image``, ``upper-left
      region``, etc.). We match case-insensitively.
    * **difficulty bin**: prose step 6 includes one of three
      distinctive phrasings (``easier than average``, ``moderate
      detection difficulty``, ``harder than average``).

    Header parsing is retained as a fast path in case a future
    training run includes the header in the target.
    """
    out = {"category": None, "spatial_descriptor": None, "difficulty_bin": None}
    text_lower = generated_text.lower()

    # ---- Header (fast path; rarely present in current training) ------
    header_match = re.search(
        r"\[category=([\w_]+),\s*scope=([\w_]+),\s*difficulty=(\w+)",
        generated_text,
    )
    if header_match:
        cat = header_match.group(1)
        if cat in CANONICAL_CATEGORIES:
            out["category"] = cat
        bin_ = header_match.group(3)
        if bin_ in ("easy", "medium", "hard"):
            out["difficulty_bin"] = bin_

    # ---- Category from prose -----------------------------------------
    if out["category"] is None:
        for cat in CANONICAL_CATEGORIES:
            cat_with_space = cat.replace("_", " ")
            # Step 4 phrases the category in several plausible ways
            # depending on model drift; check the most common ones.
            patterns = [
                f"classified as {cat}",
                f"classified as {cat_with_space}",
                f"category {cat}",
                f"category {cat_with_space}",
                f"the edit is {cat}",
                f"the edit is {cat_with_space}",
                f"is an {cat_with_space}",  # "is an object addition"
                f"is a {cat_with_space}",   # "is a style transfer"
            ]
            if any(p in text_lower for p in patterns):
                out["category"] = cat
                break

    # ---- Spatial descriptor from step 2 prose ------------------------
    # Phrase ordering matters: longer / more specific phrases first
    # so a generic phrase doesn't shadow a specific one.
    prose_to_descriptor = [
        ("spans the entire image", "whole_image"),
        ("scattered across multiple regions", "scattered"),
        ("scattered across", "scattered"),
        ("upper-left region", "upper_left"),
        ("upper left region", "upper_left"),
        ("upper-right region", "upper_right"),
        ("upper right region", "upper_right"),
        ("lower-left region", "lower_left"),
        ("lower left region", "lower_left"),
        ("lower-right region", "lower_right"),
        ("lower right region", "lower_right"),
        ("centered in the image", "centered"),
        ("alignment failure", "alignment_failed"),
        ("alignment failed", "alignment_failed"),
    ]
    for phrase, descriptor in prose_to_descriptor:
        if phrase in text_lower:
            out["spatial_descriptor"] = descriptor
            break

    # ---- Difficulty bin from step 6 prose ----------------------------
    if out["difficulty_bin"] is None:
        # Step 6 phrasings from annotator.py:
        #   "easier than average to detect, given clear local geometry..."
        #   "harder than average to detect, given diffuse geometry..."
        #   "of moderate detection difficulty"
        if "easier than average" in text_lower:
            out["difficulty_bin"] = "easy"
        elif "harder than average" in text_lower:
            out["difficulty_bin"] = "hard"
        elif ("moderate detection difficulty" in text_lower
              or "moderate difficulty" in text_lower
              or "of moderate" in text_lower):
            out["difficulty_bin"] = "medium"

    return out


def _extract_label_only_fields(generated_text: str) -> dict:
    """Parse a label-only-target generation. Expects a JSON-serialized
    triple; falls back to None on parse failure."""
    out = {"category": None, "spatial_descriptor": None, "difficulty_bin": None}
    # Try to find a JSON object in the generation.
    match = re.search(r"\{[^}]+\}", generated_text)
    if not match:
        return out
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return out
    if isinstance(parsed.get("category"), str) and parsed["category"] in CANONICAL_CATEGORIES:
        out["category"] = parsed["category"]
    if isinstance(parsed.get("spatial_descriptor"), str) and parsed["spatial_descriptor"] in CANONICAL_SPATIAL:
        out["spatial_descriptor"] = parsed["spatial_descriptor"]
    if isinstance(parsed.get("difficulty_bin"), str) and parsed["difficulty_bin"] in ("easy", "medium", "hard"):
        out["difficulty_bin"] = parsed["difficulty_bin"]
    return out


def _check_pilot_deps() -> None:
    missing = []
    for pkg in ("transformers", "peft"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"pilot evaluation requires: {missing}. Install with "
            f"`uv sync --extra pilot`."
        )


@hydra.main(
    version_base=None, config_path="../configs", config_name="pilot_evaluate",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    _check_pilot_deps()
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
    )

    # ---- load held-out evaluation set ---------------------------------
    eval_df = _load_release_annotations(
        Path(cfg.annotations_parquet), Path(cfg.image_root),
    )
    if cfg.eval_max_n is not None and cfg.eval_max_n > 0:
        eval_df = eval_df.head(int(cfg.eval_max_n))
    log.info("evaluating on %d held-out triplets", len(eval_df))

    # ---- load model + adapter -----------------------------------------
    log.info("loading base model: %s", cfg.base_model)
    processor = AutoProcessor.from_pretrained(cfg.adapter_path)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    log.info("loading LoRA adapter: %s", cfg.adapter_path)
    model = PeftModel.from_pretrained(base, cfg.adapter_path)
    model.eval()

    # ---- inference loop -----------------------------------------------
    extract = _extract_chain_fields if cfg.target_mode == "chain" else _extract_label_only_fields
    metrics_rows: list[dict] = []
    for i, rec in enumerate(eval_df.to_dict("records")):
        if i % 50 == 0:
            log.info("%d / %d", i, len(eval_df))
        real_img = _load_resized(rec["real_path"], cfg.image_max_side)
        edited_img = _load_resized(rec["edited_path"], cfg.image_max_side)
        prompt_messages = format_user_turn(
            real_img,
            edited_img,
            rec.get("instruction", ""),
            cfg.include_instruction,
        )
        text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=[[real_img, edited_img]],
            return_tensors="pt", padding=True,
        ).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
            )
        # Strip the prompt prefix from the generated tokens.
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        predicted = extract(gen_text)
        metrics_rows.append({
            "triplet_id": rec["triplet_id"],
            "true_category": rec["category"],
            "pred_category": predicted["category"],
            "true_spatial": rec["spatial_descriptor"],
            "pred_spatial": predicted["spatial_descriptor"],
            "true_bin": rec["difficulty_bin"],
            "pred_bin": predicted["difficulty_bin"],
            "category_match": predicted["category"] == rec["category"],
            "spatial_match": predicted["spatial_descriptor"] == rec["spatial_descriptor"],
            "bin_match": predicted["difficulty_bin"] == rec["difficulty_bin"],
            "generated_text": gen_text,
        })

    # ---- aggregate metrics --------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    summary = {
        "n": int(len(metrics_df)),
        "target_mode": cfg.target_mode,
        "include_instruction": bool(cfg.include_instruction),
        "category_accuracy": float(metrics_df["category_match"].mean()),
        "spatial_descriptor_accuracy": float(metrics_df["spatial_match"].mean()),
        "difficulty_bin_accuracy": float(metrics_df["bin_match"].mean()),
        # Chain faithfulness = all three structured fields match.
        # Defined for both modes; for label_only this is the natural
        # "joint accuracy" across the three predicted fields.
        "joint_field_accuracy": float(
            (metrics_df["category_match"]
             & metrics_df["spatial_match"]
             & metrics_df["bin_match"]).mean()
        ),
    }

    log.info("\n===== pilot evaluation summary =====\n%s",
             json.dumps(summary, indent=2))

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    # Per-row predictions for downstream inspection.
    rows_path = out_path.with_suffix(".per_row.parquet")
    metrics_df.to_parquet(rows_path, index=False)
    log.info("summary at %s; per-row predictions at %s", out_path, rows_path)


def _load_release_annotations(path: Path, image_root: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    root = image_root.expanduser().resolve()
    required = {
        "triplet_id", "real_path", "edited_path", "instruction",
        "reasoning_category", "reasoning_spatial_descriptor",
        "reasoning_difficulty_bin",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"release annotations missing: {sorted(missing)}")
    df = df.copy()
    df["real_path"] = df["real_path"].map(lambda p: str(root / str(p)))
    df["edited_path"] = df["edited_path"].map(lambda p: str(root / str(p)))
    df["category"] = df["reasoning_category"]
    df["spatial_descriptor"] = df["reasoning_spatial_descriptor"]
    df["difficulty_bin"] = df["reasoning_difficulty_bin"]
    return df


def _load_resized(path_str: str, max_side: int):
    from PIL import Image
    img = Image.open(path_str).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


if __name__ == "__main__":
    main()
