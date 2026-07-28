"""Pilot training script: LoRA fine-tune Qwen2-VL on EditSleuth.

Runs a single LoRA fine-tuning configuration on the
``EditSleuthCurriculumDataset``. Two target modes:

* ``chain``: the model learns to generate the full Stage~E reasoning
  chain given $(I_{\text{real}}, I_{\text{edited}}, t_{\text{instr}})$.
* ``label_only``: the model learns to generate a JSON-serialized
  $\{$category, scope, difficulty\_bin$\}$ triple, with no prose chain.

The two configurations differ only in target text; everything else
(base model, LoRA hyperparameters, optimizer, training steps) is held
fixed. This isolates the effect of the chain-as-target supervision.

This script is pilot scope: single GPU, no multi-node, no fancy
checkpointing. For full-scale training, replace the ``Trainer`` setup
with a multi-GPU framework.

Usage::

    uv run python scripts/pilot_train.py \\
        annotations_parquet=editsleuth_data/release/pico_banana_annotations.parquet \\
        image_root=/data/pico-banana-400k \\
        include_instruction=false \\
        target_mode=chain \\
        output_dir=outputs/pilot/chain_instruction_masked

Environment requirements (install with ``uv sync --extra pilot``):
    transformers >= 4.45
    peft         >= 0.12
    trl          >= 0.11
    accelerate   >= 0.34
    bitsandbytes >= 0.43  (optional, for 4-bit base model loading)
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from edit2forensics.pilot_prompt import format_user_turn

log = logging.getLogger(__name__)


def _check_pilot_deps() -> None:
    """Verify that pilot-only dependencies are installed.

    Raises a clear error if not, rather than letting an obscure
    ImportError propagate from inside the training loop.
    """
    missing = []
    for pkg in ("transformers", "peft", "trl", "accelerate"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"pilot training requires: {missing}. Install with "
            f"`uv sync --extra pilot` or `pip install {' '.join(missing)}`."
        )


def _format_chat_messages(
    real_image,
    edited_image,
    instruction: str,
    target: str,
    include_instruction: bool = True,
):
    """Build the full (user, assistant) message pair for training."""
    user_turn = format_user_turn(
        real_image, edited_image, instruction, include_instruction,
    )
    return user_turn + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        },
    ]


@hydra.main(
    version_base=None, config_path="../configs", config_name="pilot_train",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    _check_pilot_deps()
    # Imports deferred so the rest of the codebase is not coupled
    # to pilot-only dependencies.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
        TrainingArguments,
    )
    from trl import SFTTrainer

    from edit2forensics.curriculum.dataset import (
        CurriculumDatasetConfig,
        EditSleuthCurriculumDataset,
    )

    # ---- dataset ------------------------------------------------------
    ds_config = CurriculumDatasetConfig(
        target_mode=cfg.target_mode,
        samples_per_category_per_bin=cfg.samples_per_category_per_bin,
        seed=cfg.seed,
        image_max_side=cfg.image_max_side,
        include_instruction=cfg.include_instruction,
        redact_instruction_target=cfg.redact_instruction_target,
    )
    train_ds = EditSleuthCurriculumDataset(
        annotations_parquet=Path(cfg.annotations_parquet),
        image_root=Path(cfg.image_root),
        config=ds_config,
    )
    log.info("training dataset: %d examples", len(train_ds))

    # ---- base model + LoRA --------------------------------------------
    log.info("loading base model: %s", cfg.base_model)
    processor = AutoProcessor.from_pretrained(cfg.base_model)
    # Use LEFT padding for training. Right-padding has produced
    # backward-pass NaN gradients on Qwen2-VL in our environment
    # despite the attention mask being correctly applied — the
    # gradient flow through right-padded attention positions
    # interacts unfavorably with bf16 numerics. Left-padding is
    # the standard fix for this class of bug; it does not change
    # the actual training signal because the loss is masked at
    # padding positions either way.
    processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
    log.info("processor padding_side set to: %s",
             processor.tokenizer.padding_side)

    # Respect the bf16 config flag for the model's compute dtype.
    model_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    log.info("loading model with dtype=%s", model_dtype)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.base_model,
        torch_dtype=model_dtype,
        device_map="auto",
    )
    # PEFT's target_modules accepts two formats:
    #   - list of strings → substring match (every parameter name
    #     containing any string is wrapped).
    #   - single regex string → full pattern match.
    # We accept both via the YAML config. Hydra deserializes a YAML
    # list as a Python list and a YAML string as a Python str.
    raw_targets = cfg.lora_target_modules
    if isinstance(raw_targets, str):
        target_modules_arg = raw_targets
    else:
        target_modules_arg = list(raw_targets)
    log.info(
        "LoRA target_modules: %r (type=%s)",
        target_modules_arg, type(target_modules_arg).__name__,
    )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules_arg,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Diagnostic: enumerate which modules LoRA actually wrapped.
    # When ``target_modules`` is a string list (e.g. ["q_proj",
    # "v_proj"]), PEFT does substring matching against every
    # parameter name in the model. For multi-modal models like
    # Qwen2-VL with both a vision encoder and a language model,
    # this can wrap LoRA around BOTH the vision encoder's attention
    # projections AND the language model's. If the vision
    # encoder's bf16 backward is unstable, gradients through it
    # produce NaN even though the language model is fine. We print
    # the wrapped modules here so any unintended targeting is
    # visible at startup.
    wrapped_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            wrapped_modules.append(name)
    log.info(
        "LoRA wrapped %d modules; first 8 names:\n  %s",
        len(wrapped_modules),
        "\n  ".join(wrapped_modules[:8]),
    )
    # Detect vision-encoder targeting and warn if it's happening.
    vision_wrapped = [n for n in wrapped_modules
                      if "visual" in n.lower() or "vision" in n.lower()]
    if vision_wrapped:
        log.warning(
            "LoRA is wrapping %d vision-encoder modules (e.g. %s). "
            "On Qwen2-VL with bf16, gradients through the vision encoder "
            "can produce NaN. Consider restricting target_modules to the "
            "language model only.",
            len(vision_wrapped), vision_wrapped[0],
        )

    # When gradient_checkpointing is on, LoRA gradients don't flow
    # into the frozen base model unless we explicitly require input
    # grads. This is a no-op when checkpointing is off, so it's
    # safe to call unconditionally.
    if cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        log.info("enabled input_require_grads for gradient checkpointing + LoRA")

    # ---- pre-compute the assistant-turn marker token sequence -------
    # Qwen2-VL's chat template emits ``<|im_start|>assistant\n`` to
    # mark the start of the assistant turn. We tokenize this once and
    # search for it in each row's input_ids during collation. This is
    # more robust than measuring prompt length via a separate
    # tokenization pass: it works regardless of how the processor
    # expands image tokens, regardless of padding direction, and
    # regardless of any subtle differences between batched and single-
    # example processor calls.
    assistant_marker_text = "<|im_start|>assistant\n"
    assistant_marker_ids = processor.tokenizer.encode(
        assistant_marker_text, add_special_tokens=False,
    )
    if not assistant_marker_ids:
        raise RuntimeError(
            "tokenizer produced empty token sequence for the assistant "
            "marker '<|im_start|>assistant\\n'; chat template may have "
            "diverged from the expected Qwen2-VL format."
        )
    log.info(
        "assistant marker token ids: %s (length %d)",
        assistant_marker_ids, len(assistant_marker_ids),
    )

    def _find_marker_end(row_ids, marker_ids):
        """Return the index immediately after the marker subsequence
        in ``row_ids``. Returns -1 if the marker isn't found."""
        m = len(marker_ids)
        n = len(row_ids)
        if m > n:
            return -1
        for start in range(n - m + 1):
            if all(int(row_ids[start + j]) == marker_ids[j] for j in range(m)):
                return start + m
        return -1

    # ---- chat-template adapter ---------------------------------------
    def _data_collator(features: list[dict]) -> dict:
        """Apply Qwen2-VL chat template, tokenize, and mask the loss
        so it only covers the assistant-turn (chain or label-only)
        target tokens.

        Without masking, the cross-entropy loss includes user-turn
        text and image-placeholder tokens. The image-placeholder
        positions in particular drive loss to NaN because the model
        has no meaningful target at those positions.

        Masking algorithm:
          1. Tokenize the full (user + assistant) conversation as a
             padded batch — this is what we feed to the model.
          2. For each row, search input_ids for the
             ``<|im_start|>assistant\\n`` token sequence; mask
             [0, marker_end) so the loss only covers the chain text
             (and the trailing ``<|im_end|>``).
          3. Mask any padding tokens.

        This marker-search approach replaces an earlier prompt-
        length-measurement approach which produced incorrect masks
        when the processor's batched-with-padding tokenization
        differed slightly from its single-example tokenization
        (image-token leakage into the active region).
        """
        full_messages_list = [
            _format_chat_messages(
                f["real_image"], f["edited_image"],
                f["instruction"], f["target"], cfg.include_instruction,
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

        # Build labels: copy input_ids, then mask out (a) all positions
        # up to and including the assistant marker, (b) padding tokens.
        labels = batch["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id

        n_rows = batch["input_ids"].shape[0]
        markers_not_found = []
        for i in range(n_rows):
            row_ids = batch["input_ids"][i].tolist()
            marker_end = _find_marker_end(row_ids, assistant_marker_ids)
            if marker_end == -1:
                markers_not_found.append(i)
                # Defensively mask the entire row so this example does
                # not contribute random gradient. The sanity check
                # below will surface this loud and clear.
                labels[i, :] = -100
            else:
                labels[i, :marker_end] = -100

        if pad_id is not None:
            labels[batch["input_ids"] == pad_id] = -100

        if markers_not_found:
            raise RuntimeError(
                f"assistant marker not found in {len(markers_not_found)} "
                f"row(s) of a batch (indices {markers_not_found}); chat "
                f"template tokenization diverged from expectations. "
                f"Investigate by running scripts/inspect_pilot_batch.py."
            )

        # Sanity check: if every label is masked the loss will be
        # NaN. Surface this rather than letting NaN propagate.
        n_active_per_row = (labels != -100).sum(dim=1)
        if (n_active_per_row == 0).any():
            log.warning(
                "collator produced row(s) with all labels masked; "
                "n_active_per_row=%s", n_active_per_row.tolist(),
            )

        batch["labels"] = labels
        return batch

    # ---- trainer ------------------------------------------------------
    # Use SFTConfig (TRL's subclass of TrainingArguments) when
    # available — it exposes ``dataset_kwargs={"skip_prepare_dataset":
    # True}``, which prevents TRL from trying to apply its own
    # chat-template formatting to our custom multi-image Dataset.
    # That auto-prep path expects column names like ``messages``/
    # ``text``/``prompt`` and triggers the ``column_names``
    # AttributeError on plain torch Datasets in older TRL versions.
    #
    # We deliberately do NOT pass max_seq_length / max_length: the
    # field name has shifted across TRL versions (max_seq_length in
    # <=0.10, max_length in 0.11, removed in some 0.12+ branches),
    # and our chains are short enough (~80-150 words = ~120-300 tokens)
    # that the model's default context window (32K for Qwen2-VL-2B)
    # provides ample headroom even with two 448px images.
    #
    # ``gradient_checkpointing`` defaults to False because it has
    # produced bf16 gradient-NaN on Qwen2-VL-2B + LoRA in our
    # environment: step 1 of training produces a finite loss but
    # NaN gradient norm, after which the LoRA weights are corrupted
    # and every subsequent step has loss=0. Disabling checkpointing
    # avoids the bf16 numerical instability at the cost of higher
    # VRAM (still fits comfortably on a 24GB card for our pilot).
    try:
        from trl import SFTConfig
        args = SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            bf16=cfg.bf16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            remove_unused_columns=False,
            report_to=cfg.report_to,
            dataset_kwargs={"skip_prepare_dataset": True},
        )
    except ImportError:
        args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            bf16=cfg.bf16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            remove_unused_columns=False,
            report_to=cfg.report_to,
        )
    # SFTTrainer's tokenizer-passing argument was renamed:
    # ``tokenizer=`` in older TRL (<=0.10ish), ``processing_class=``
    # in current TRL. Try the new name first, fall back to the old.
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=_data_collator,
    )
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=processor.tokenizer)
    except TypeError:
        log.info("falling back to legacy SFTTrainer(tokenizer=...) argument")
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=processor.tokenizer)
    log.info("starting training")
    trainer.train()
    log.info("training complete; saving adapter to %s", cfg.output_dir)
    model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
