#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 RELEASE_ROOT PICO_ROOT MAGICBRUSH_ROOT OUTPUT_ROOT"
  exit 2
fi

release_root="$(cd "$1" && pwd)"
pico_root="$(cd "$2" && pwd)"
magicbrush_root="$(cd "$3" && pwd)"
output_root="$4"
visible_adapter="$output_root/visible_label_only"
mkdir -p "$output_root"

if [[ ! -f "$visible_adapter/adapter_config.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/pilot_train.py \
    annotations_parquet="$release_root/pico_banana_annotations.parquet" \
    image_root="$pico_root" \
    include_instruction=true \
    target_mode=label_only \
    output_dir="$visible_adapter"
fi

CUDA_VISIBLE_DEVICES=0 uv run python scripts/generate_label_saliency.py \
  --annotations "$release_root/magicbrush_dev_annotations.parquet" \
  --image-root "$magicbrush_root" \
  --adapter-path "$visible_adapter" \
  --output-dir "$output_root/label_only_saliency" \
  --n 50 \
  --include-instruction &
saliency_pid=$!

CUDA_VISIBLE_DEVICES=1 uv run python scripts/generate_freeform_vlm_chains.py \
  --annotations "$release_root/pico_banana_annotations.parquet" \
  --image-root "$pico_root" \
  --output "$output_root/freeform_vlm_chains_200.parquet" \
  --n 200 \
  --seed 2026 &
freeform_pid=$!

wait "$saliency_pid"
wait "$freeform_pid"

uv run python scripts/audit_human_difficulty.py prepare \
  --input "$release_root/pico_banana_annotations.parquet" \
  --image-root "$pico_root" \
  --bundle-dir "$output_root/human_difficulty_audit_100" \
  --n 100 \
  --annotators 2 \
  --seed 2026

uv run python scripts/prepare_paired_chain_audit.py prepare \
  --input "$output_root/freeform_vlm_chains_200.parquet" \
  --image-root "$pico_root" \
  --bundle-dir "$output_root/paired_chain_audit_200" \
  --annotators 2 \
  --seed 2026

echo "Remaining rebuttal experiment artifacts are ready under $output_root"
