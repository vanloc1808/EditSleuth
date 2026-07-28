#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 PICO_BANANA_ROOT MAGICBRUSH_ROOT OUTPUT_ROOT"
  exit 2
fi

pico_root="$(cd "$1" && pwd)"
magicbrush_root="$(cd "$2" && pwd)"
output_root="$3"
release_root="${EDITSLEUTH_RELEASE_ROOT:-editsleuth_data/release}"

mkdir -p "$output_root"

CUDA_VISIBLE_DEVICES=0 uv run python scripts/pilot_train.py \
  annotations_parquet="$release_root/pico_banana_annotations.parquet" \
  image_root="$pico_root" \
  include_instruction=false \
  redact_instruction_target=true \
  target_mode=chain \
  output_dir="$output_root/chain_instruction_masked" &
chain_pid=$!

CUDA_VISIBLE_DEVICES=1 uv run python scripts/pilot_train.py \
  annotations_parquet="$release_root/pico_banana_annotations.parquet" \
  image_root="$pico_root" \
  include_instruction=false \
  target_mode=label_only \
  output_dir="$output_root/label_only_instruction_masked" &
label_pid=$!

wait "$chain_pid"
wait "$label_pid"

CUDA_VISIBLE_DEVICES=2 uv run python scripts/pilot_evaluate.py \
  annotations_parquet="$release_root/magicbrush_dev_annotations.parquet" \
  image_root="$magicbrush_root" \
  adapter_path="$output_root/chain_instruction_masked" \
  include_instruction=false \
  target_mode=chain \
  output_path="$output_root/chain_instruction_masked_eval.json"

CUDA_VISIBLE_DEVICES=2 uv run python scripts/pilot_evaluate.py \
  annotations_parquet="$release_root/magicbrush_dev_annotations.parquet" \
  image_root="$magicbrush_root" \
  adapter_path="$output_root/label_only_instruction_masked" \
  include_instruction=false \
  target_mode=label_only \
  output_path="$output_root/label_only_instruction_masked_eval.json"
