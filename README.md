# EditSleuth

A dataset of grounded reasoning chains for image-edit forensics. Re-purposes existing image-editing triplets (real image, edited image, instruction) as forensic-detection training data, with masks, difficulty scores, category labels, and six-step reasoning chains composed deterministically from upstream artifacts.

## Project layout

```
src/edit2forensics/      Pipeline source code (stages A–E).
scripts/                 Command-line entry points (one per stage).
configs/                 Hydra configs, one per script.
tests/                   Test suite (370 tests).
docs/                    Runbooks and troubleshooting.
```

Default output root: `$E2F_OUTPUT_ROOT` (or `/path/to/editsleuth_output`). Override per-run with `paths.output_root=...`.

## Install

```bash
uv sync
uv run pytest tests/    # 370 passing
```

## Pipeline

Each stage reads parquet input and writes parquet output. Stages A–E run sequentially; D depends on A only; E depends on A, B, C, D.

### Stage A — Ingest

Convert source-dataset records into canonical `EditTriplet` parquet.

```bash
uv run python scripts/ingest.py \
    adapter=pico_banana \
    paths.dataset_root=/path/to/pico-banana-400k
```

Output: `${output_root}/triplets/<dataset>.parquet`. Built-in adapters: `pico_banana`, `magicbrush`.

### Stage B — Mask generation

Compute a binary edit mask for each triplet from the (real, edited) pair.

```bash
uv run python scripts/generate_masks.py \
    triplets_path=${output_root}/triplets/pico_banana.parquet \
    mask_generator=lab_plus_lpips_plus_ssim \
    mask_generator.signals.1.device=cuda \
    run_tag=pico_banana_lab_lpips_ssim # example [you given tag name]
```

Output: `${output_root}/masks/<dataset>/` (one PNG per triplet) plus `${output_root}/masks/<dataset>.parquet` (mask-statistics index).

Threshold calibration (run once per signal stack):

```bash
uv run python scripts/sweep_global_threshold.py \
    paths.triplets_parquet=${output_root}/triplets/pico_banana.parquet
```

### Stage C — Difficulty scoring

Compute per-triplet difficulty scores from masks and instructions. V2 formula (3-component, default).

```bash
uv run python scripts/score_difficulty.py \
    triplets_path=${output_root}/triplets/pico_banana.parquet \
    mask_artifacts_path=${output_root}/masks/pico_banana.parquet \
    run_tag=given_name \
    difficulty_scorer=v2_default # for v2
```

Output: `${output_root}/difficulty/<dataset>.parquet`.

If you regenerate masks, also re-run mask compactness:

```bash
uv run python scripts/add_mask_compactness.py \
    mask_artifacts_path=${output_root}/masks/pico_banana.parquet \
    output_path=/your/output/path/name.parquet
```

### Stage D — Category classification

Assign a category label from the 12-class taxonomy via rules.

```bash
uv run python scripts/classify_categories.py \
    triplets_path=${output_root}/triplets/pico_banana.parquet \
    run_tag=given_name
```

Output: `${output_root}/category/<dataset>.parquet`. Coverage: 100% on Pico-Banana (uses source labels), ~56% on MagicBrush (rule-based, 44% fall back to `other`).

### Stage E — Reasoning chain composition

Compose six-step grounded reasoning chains from upstream artifacts.

```bash
uv run python scripts/annotate_reasoning.py \
    triplets_path=${output_root}/triplets/pico_banana.parquet \
    mask_artifacts_path=${output_root}/masks/pico_banana.parquet \
    difficulty_path=${output_root}/difficulty/pico_banana.parquet \
    category_path=${output_root}/category/pico_banana.parquet \
    run_tag=given_name
```

Output: `${output_root}/reasoning/<dataset>.parquet`. One chain per triplet plus a structured header.

## Pilot training

LoRA fine-tune Qwen2-VL-2B-Instruct on EditSleuth. Two target modes: `chain` (generate the full six-step chain) and `label_only` (generate a JSON triple of category, spatial descriptor, difficulty bin).

```bash
uv run python scripts/pilot_train.py \
    triplets_parquet=${output_root}/triplets/pico_banana.parquet \
    reasoning_parquet=${output_root}/reasoning/pico_banana.parquet \
    target_mode=chain \
    output_dir=${output_root}/pilot/chain_run
```

Defaults: `bf16=false` (fp32, required for backward-pass stability on Qwen2-VL + PEFT — see `docs/pilot_quickstart.md`), `gradient_checkpointing=false`, LoRA on language-model `q_proj`/`v_proj` only, left-padding. ~10 hours per config on a single H100.

For label-only training, set `target_mode=label_only` and a separate `output_dir`.

## Pilot evaluation

Greedy-decoded inference + regex extraction of structured fields.

```bash
uv run python scripts/pilot_evaluate.py \
    triplets_parquet=${output_root}/triplets/magicbrush_dev.parquet \
    reasoning_parquet=${output_root}/reasoning/magicbrush_dev.parquet \
    adapter_path=${output_root}/pilot/chain_run \
    target_mode=chain \
    output_path=${output_root}/pilot/chain_eval.json
```

Outputs:
- `chain_eval.json` — summary metrics (per-field accuracy, extraction recall, joint accuracy).
- `chain_eval.per_row.parquet` — per-triplet predictions and extracted fields.

To re-run extraction after editing the extractor (no inference cost):

```bash
uv run python scripts/reextract_pilot_fields.py \
    --predictions-parquet ${output_root}/pilot/chain_eval.per_row.parquet \
    --target-mode chain \
    --output-summary ${output_root}/pilot/chain_eval_v2.json
```

To inspect a few rows of predictions:

```bash
uv run python scripts/inspect_pilot_predictions.py \
    --predictions-parquet ${output_root}/pilot/chain_eval.per_row.parquet \
    --n 5
```

## Tests

```bash
uv run pytest tests/        # 370 tests, ~15 sec
uv run pytest tests/ -v     # verbose
```

## Documentation

- `docs/pilot_quickstart.md` — pilot training/eval runbook + troubleshooting.
