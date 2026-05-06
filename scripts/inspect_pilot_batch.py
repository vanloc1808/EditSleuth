"""Inspect what the pilot training collator produces.

Builds the first training batch through the same collator that
``pilot_train.py`` uses, then prints diagnostics about the masking:

* Which token positions are active (label != -100) vs masked.
* The decoded text at the active positions, to verify it matches
  the expected assistant target (the reasoning chain or the JSON
  label triple).
* Per-row active-token counts.

This is a smoke-test tool. Run it before launching a full pilot
training to verify the masking algorithm is producing the labels we
expect. If the active-token decoded text doesn't match the chain,
training will produce NaN loss or memorize the prompt — both of which
have been observed in early pilot runs.

Usage::

    uv run python scripts/inspect_pilot_batch.py \\
        triplets_parquet=artifacts/triplets/pico_banana.parquet \\
        reasoning_parquet=artifacts/reasoning/pico_banana.parquet \\
        target_mode=chain
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs", config_name="pilot_train",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Defer imports so the rest of the codebase doesn't need pilot deps.
    from transformers import AutoProcessor

    from edit2forensics.curriculum.dataset import (
        CurriculumDatasetConfig,
        EditSleuthCurriculumDataset,
    )
    # Import the helpers that pilot_train uses, to ensure exactly the
    # same chat-template formatting as during training.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pilot_train import _format_chat_messages

    # ---- build dataset (small, just enough for a few examples) -------
    ds_config = CurriculumDatasetConfig(
        target_mode=cfg.target_mode,
        samples_per_category_per_bin=2,   # tiny for inspection only
        seed=cfg.seed,
        image_max_side=cfg.image_max_side,
    )
    train_ds = EditSleuthCurriculumDataset(
        triplets_parquet=Path(cfg.triplets_parquet),
        reasoning_parquet=Path(cfg.reasoning_parquet),
        config=ds_config,
    )
    log.info("dataset has %d examples; inspecting first batch", len(train_ds))

    # Take the first 2 examples as a "batch."
    batch_size = 2
    features = [train_ds[i] for i in range(min(batch_size, len(train_ds)))]

    # ---- run the same collator logic as pilot_train.py --------------
    processor = AutoProcessor.from_pretrained(cfg.base_model)
    # Match pilot_train.py: use left-padding to avoid the
    # right-padding-induced backward NaN on Qwen2-VL.
    processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"

    # Pre-compute the assistant-turn marker token sequence (must
    # match exactly what pilot_train.py uses).
    assistant_marker_text = "<|im_start|>assistant\n"
    assistant_marker_ids = processor.tokenizer.encode(
        assistant_marker_text, add_special_tokens=False,
    )
    print(f"assistant marker token ids: {assistant_marker_ids} "
          f"(length {len(assistant_marker_ids)})\n")

    def _find_marker_end(row_ids, marker_ids):
        m = len(marker_ids)
        n = len(row_ids)
        if m > n:
            return -1
        for start in range(n - m + 1):
            if all(int(row_ids[start + j]) == marker_ids[j] for j in range(m)):
                return start + m
        return -1

    full_messages_list = [
        _format_chat_messages(
            f["real_image"], f["edited_image"],
            f["instruction"], f["target"],
        )
        for f in features
    ]
    full_texts = [
        processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=False,
        )
        for m in full_messages_list
    ]
    images_per_example = [
        [f["real_image"], f["edited_image"]] for f in features
    ]

    batch = processor(
        text=full_texts,
        images=images_per_example,
        padding=True,
        return_tensors="pt",
    )
    log.info("full batch input_ids shape: %s", tuple(batch["input_ids"].shape))

    # Build labels via marker-search (same as the pilot_train.py
    # collator).
    labels = batch["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    n_rows = batch["input_ids"].shape[0]
    marker_ends = []
    for i in range(n_rows):
        row_ids = batch["input_ids"][i].tolist()
        end = _find_marker_end(row_ids, assistant_marker_ids)
        marker_ends.append(end)
        if end == -1:
            print(f"WARNING: row {i} — assistant marker NOT FOUND. "
                  f"Chat template may have diverged from expected format.")
            labels[i, :] = -100
        else:
            labels[i, :end] = -100
    if pad_id is not None:
        labels[batch["input_ids"] == pad_id] = -100
    print(f"per-row marker_end positions: {marker_ends}\n")

    # ---- diagnostics -------------------------------------------------
    print("\n===== batch label masking diagnostics =====\n")
    for i, f in enumerate(features):
        print(f"--- example {i}: triplet_id={f['triplet_id']} ---")
        n_total = labels.shape[1]
        n_masked = int((labels[i] == -100).sum())
        n_active = n_total - n_masked
        print(f"  total tokens:    {n_total}")
        print(f"  masked tokens:   {n_masked}  ({100*n_masked/n_total:.1f}%)")
        print(f"  active (loss):   {n_active}  ({100*n_active/n_total:.1f}%)")

        # Decode the active-token positions to verify they're the
        # assistant target.
        active_token_ids = batch["input_ids"][i, labels[i] != -100]
        decoded = processor.tokenizer.decode(
            active_token_ids, skip_special_tokens=False,
        )
        print(f"  decoded active text:\n    {decoded[:500]}")
        if len(decoded) > 500:
            print(f"    ... ({len(decoded) - 500} more chars)")
        print(f"  expected target (from dataset):\n    {f['target'][:500]}")
        if len(f['target']) > 500:
            print(f"    ... ({len(f['target']) - 500} more chars)")
        print()

    # Sanity check the sanity check.
    if any((labels == -100).all(dim=1)):
        print("WARNING: at least one row has ALL labels masked — loss "
              "would be NaN. Investigate before launching training.")
    else:
        print("OK: every row has at least one active label position.")

    # ---- comprehensive batch-tensor diagnostics ---------------------
    # The masking algorithm is verified above. If training is still
    # failing (e.g., NaN gradients on step 1), the cause may be in
    # what the processor returned in the batch beyond input_ids and
    # labels. Surface every key, its shape, dtype, and statistics.
    print("\n===== batch-tensor diagnostics =====\n")
    import torch
    print(f"batch keys: {sorted(batch.keys())}")
    for key in sorted(batch.keys()):
        val = batch[key]
        if not isinstance(val, torch.Tensor):
            print(f"  {key}: type={type(val).__name__}, value={val!r}")
            continue
        info = (f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
        if val.dtype.is_floating_point:
            n_nan = int(torch.isnan(val).sum())
            n_inf = int(torch.isinf(val).sum())
            v_min = float(val.min())
            v_max = float(val.max())
            info += (f"\n      n_nan={n_nan}, n_inf={n_inf}, "
                     f"min={v_min:.4f}, max={v_max:.4f}")
        elif val.dtype in (torch.int64, torch.int32, torch.long):
            v_min = int(val.min())
            v_max = int(val.max())
            n_unique = int(val.unique().numel())
            info += (f"\n      min={v_min}, max={v_max}, n_unique={n_unique}")
        print(info)

    # Specifically check attention_mask if present.
    if "attention_mask" in batch:
        am = batch["attention_mask"]
        n_zeros_per_row = (am == 0).sum(dim=1).tolist()
        n_total_per_row = am.shape[1]
        print(f"\nattention_mask: per-row padded-position counts: {n_zeros_per_row}")
        print(f"   sequence length: {n_total_per_row}")
        # Confirm shape matches input_ids.
        if am.shape != batch["input_ids"].shape:
            print(f"WARNING: attention_mask shape {tuple(am.shape)} does "
                  f"NOT match input_ids shape {tuple(batch['input_ids'].shape)}!")

        # Detect which side padding is on.
        # Right-padding: zeros at the end. Left-padding: zeros at the start.
        for i in range(am.shape[0]):
            if n_zeros_per_row[i] == 0:
                continue
            row = am[i]
            zeros_at_start = int((row[:n_zeros_per_row[i]] == 0).all())
            zeros_at_end = int((row[-n_zeros_per_row[i]:] == 0).all())
            side = ("left" if zeros_at_start else
                    "right" if zeros_at_end else
                    "scattered")
            print(f"   row {i}: padding_side={side}")
    else:
        print("\nWARNING: batch has no attention_mask key. Without an "
              "explicit attention mask, the model attends to padded "
              "positions, which can produce NaN gradients on the "
              "backward pass even if forward looks fine.")

    # Verify mm_token_type_ids alignment with image-pad tokens.
    # mm_token_type_ids is a Qwen2-VL tensor that flags which input_id
    # positions are image-pad placeholders (=1) vs text (=0). The two
    # must be consistent: every position with input_ids == 151655
    # (image_pad) should have mm_token_type_ids == 1, and vice versa.
    # Misalignment here would route an image embedding through the
    # text path or vice versa, with possibly NaN-prone results.
    if "mm_token_type_ids" in batch:
        IMAGE_PAD_ID = 151655
        ids = batch["input_ids"]
        types = batch["mm_token_type_ids"]
        is_image_pad = (ids == IMAGE_PAD_ID)
        type_marks_image = (types == 1)
        mismatches = int((is_image_pad != type_marks_image).sum())
        print(f"\nmm_token_type_ids alignment with image_pad input_ids: "
              f"{'consistent' if mismatches == 0 else f'{mismatches} mismatches'}")
        if mismatches > 0:
            print(f"  WARNING: {mismatches} positions disagree between "
                  f"input_ids and mm_token_type_ids — likely a bug.")


if __name__ == "__main__":
    main()
