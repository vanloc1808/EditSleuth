# EditSleuth Pilot Fine-Tuning Quickstart

This document walks through running the pilot fine-tuning study
end-to-end. The pilot has two configurations (chain target vs
label-only target) trained on the same data, evaluated on the same
held-out set, with the same hyperparameters. The configurations differ
only in the supervised target — this isolates the effect of the
chain-as-target supervision.

## Prerequisites

The pilot uses HuggingFace Transformers + PEFT + TRL. Install the
optional pilot dependency group:

```bash
uv sync --extra pilot
```

If `pyproject.toml` does not yet have a `pilot` extra, add this to
`[project.optional-dependencies]`:

```toml
pilot = [
    "torch>=2.4.0",
    "transformers>=4.45.0",
    "peft>=0.12.0",
    "trl>=0.11.0",
    "accelerate>=0.34.0",
    "bitsandbytes>=0.43.0",
]
```

A single A100 40GB or H100 is sufficient for Qwen2-VL-2B + LoRA at
batch size 2 with gradient accumulation 8. A 24GB consumer card
(e.g. RTX 4090) also works after lowering `image_max_side` to 336.

## Step 1: Verify upstream artifacts exist

The pilot reads:

- `artifacts/triplets/pico_banana.parquet` (Stage A output)
- `artifacts/reasoning/pico_banana.parquet` (Stage E output)
- `artifacts/triplets/magicbrush_dev.parquet` (held-out evaluation)
- `artifacts/reasoning/magicbrush_dev.parquet` (held-out evaluation)

Confirm all four exist before launching training.

## Step 2: Train the chain-target model

```bash
uv run python scripts/pilot_train.py \
    triplets_parquet=artifacts/triplets/pico_banana.parquet \
    reasoning_parquet=artifacts/reasoning/pico_banana.parquet \
    target_mode=chain \
    output_dir=artifacts/pilot/chain_run
```

Default hyperparameters: 1 epoch over ~19,800 stratified training
examples, LoRA $r=16$, learning rate $1\mathrm{e}{-4}$. Estimated wall
time on a single A100: 4-6 hours.

The script saves the trained LoRA adapter to
`artifacts/pilot/chain_run/`.

## Step 3: Train the label-only baseline

```bash
uv run python scripts/pilot_train.py \
    triplets_parquet=artifacts/triplets/pico_banana.parquet \
    reasoning_parquet=artifacts/reasoning/pico_banana.parquet \
    target_mode=label_only \
    output_dir=artifacts/pilot/label_only_run
```

Same hyperparameters, only the target string differs. The output
adapter is saved to `artifacts/pilot/label_only_run/`. Same estimated
wall time as Step 2.

## Step 4: Evaluate both adapters on MagicBrush dev

```bash
# Chain-target evaluation
uv run python scripts/pilot_evaluate.py \
    triplets_parquet=artifacts/triplets/magicbrush_dev.parquet \
    reasoning_parquet=artifacts/reasoning/magicbrush_dev.parquet \
    adapter_path=artifacts/pilot/chain_run \
    target_mode=chain \
    output_path=artifacts/pilot/chain_eval.json

# Label-only evaluation
uv run python scripts/pilot_evaluate.py \
    triplets_parquet=artifacts/triplets/magicbrush_dev.parquet \
    reasoning_parquet=artifacts/reasoning/magicbrush_dev.parquet \
    adapter_path=artifacts/pilot/label_only_run \
    target_mode=label_only \
    output_path=artifacts/pilot/label_only_eval.json
```

Each evaluation runs greedy decoding over 528 MagicBrush dev triplets;
estimated wall time ~30-45 minutes per adapter.

The output JSON contains four metrics:
- `category_accuracy`: top-1 match between predicted and true category.
- `spatial_descriptor_accuracy`: top-1 match for the spatial descriptor.
- `difficulty_bin_accuracy`: top-1 match for the difficulty bin.
- `joint_field_accuracy`: all three structured fields match
  simultaneously (chain faithfulness for the chain-target variant;
  full-label-correct rate for the label-only variant).

A per-row predictions parquet is also saved at
`artifacts/pilot/<name>_eval.per_row.parquet` for inspection of
specific failure cases.

## Quick smoke test (10 minutes, no GPU)

To verify the pipeline scripts run end-to-end without committing to
a full training run, you can launch the smallest possible
configuration:

```bash
# 100 training examples instead of 19,800
uv run python scripts/pilot_train.py \
    triplets_parquet=artifacts/triplets/pico_banana.parquet \
    reasoning_parquet=artifacts/reasoning/pico_banana.parquet \
    target_mode=chain \
    samples_per_category_per_bin=3 \
    num_train_epochs=1 \
    output_dir=/tmp/pilot_smoke
```

This trains on ~99 examples in roughly 5-10 minutes on a small GPU.
The resulting adapter will not have learned anything substantive but
the pipeline will have run end-to-end, validating that the data
loading, chat templating, and training-loop integration work
correctly before launching the full pilot.

## Troubleshooting

- **Out of memory during training.** Lower `per_device_train_batch_size`
  to 1, raise `gradient_accumulation_steps` to 16 to keep effective
  batch size constant. Or lower `image_max_side` to 336.
- **NaN losses early in training.** Often caused by mixed-precision
  instability with very high learning rates. Lower `learning_rate`
  to 5e-5.
- **`Qwen2VLForConditionalGeneration` not found.** Ensure
  `transformers >= 4.45`. Earlier versions don't include the Qwen2-VL
  modeling code.
- **Evaluation generates empty strings.** Check that
  `processor.apply_chat_template(..., add_generation_prompt=True)` is
  producing a prompt with a trailing assistant-role marker. If not,
  the model has nothing to generate from.
