#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

trap 'status=$?; log "ERROR: launcher stopped at line $LINENO (exit $status)"; exit "$status"' ERR

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

log "Starting remaining rebuttal workflow"
log "Release data: $release_root"
log "Pico-Banana images: $pico_root"
log "MagicBrush images: $magicbrush_root"
log "Output directory: $output_root"

if [[ ! -f "$visible_adapter/adapter_config.json" ]]; then
  log "STEP 1/4: visible label-only adapter is absent; training it on GPU 0"
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/pilot_train.py \
    annotations_parquet="$release_root/pico_banana_annotations.parquet" \
    image_root="$pico_root" \
    include_instruction=true \
    target_mode=label_only \
    output_dir="$visible_adapter"
  log "STEP 1/4 complete: adapter saved to $visible_adapter"
else
  log "STEP 1/4 skipped: reusing adapter at $visible_adapter"
fi

saliency_pid=""
freeform_pid=""
saliency_status=0
freeform_status=0

if [[ -f "$output_root/label_only_saliency/saliency_index.parquet" ]]; then
  log "STEP 2/4 skipped: saliency index already exists"
else
  log "STEP 2/4: starting 50-example label saliency job on GPU 0"
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/generate_label_saliency.py \
    --annotations "$release_root/magicbrush_dev_annotations.parquet" \
    --image-root "$magicbrush_root" \
    --adapter-path "$visible_adapter" \
    --output-dir "$output_root/label_only_saliency" \
    --n 50 \
    --include-instruction &
  saliency_pid=$!
  log "STEP 2/4 saliency process started (PID $saliency_pid)"
fi

if [[ -f "$output_root/freeform_vlm_chains_200.parquet" ]]; then
  log "STEP 3/4 skipped: 200-chain free-form baseline already exists"
else
  log "STEP 3/4: starting 200-chain free-form baseline on GPU 1"
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/generate_freeform_vlm_chains.py \
    --annotations "$release_root/pico_banana_annotations.parquet" \
    --image-root "$pico_root" \
    --output "$output_root/freeform_vlm_chains_200.parquet" \
    --n 200 \
    --seed 2026 &
  freeform_pid=$!
  log "STEP 3/4 free-form process started (PID $freeform_pid)"
fi

if [[ -n "$saliency_pid" ]]; then
  log "Waiting for saliency process PID $saliency_pid"
  if wait "$saliency_pid"; then
    log "STEP 2/4 complete: saliency artifacts are ready"
  else
    saliency_status=$?
    log "ERROR: saliency process failed with exit $saliency_status"
  fi
fi

if [[ -n "$freeform_pid" ]]; then
  log "Waiting for free-form process PID $freeform_pid"
  if wait "$freeform_pid"; then
    log "STEP 3/4 complete: free-form baseline is ready"
  else
    freeform_status=$?
    log "ERROR: free-form process failed with exit $freeform_status"
  fi
fi

if (( saliency_status != 0 || freeform_status != 0 )); then
  log "Stopping before audit preparation because a GPU job failed"
  exit 1
fi

log "STEP 4/4: preparing the 100-triplet human difficulty bundle"
uv run python scripts/audit_human_difficulty.py prepare \
  --input "$release_root/pico_banana_annotations.parquet" \
  --image-root "$pico_root" \
  --bundle-dir "$output_root/human_difficulty_audit_100" \
  --n 100 \
  --annotators 2 \
  --seed 2026

log "STEP 4/4: preparing the paired 200-chain audit bundle"
uv run python scripts/prepare_paired_chain_audit.py prepare \
  --input "$output_root/freeform_vlm_chains_200.parquet" \
  --image-root "$pico_root" \
  --bundle-dir "$output_root/paired_chain_audit_200" \
  --annotators 2 \
  --seed 2026

log "STEP 4/4 complete: both human-audit bundles are ready"
log "Remaining rebuttal experiment artifacts are ready under $output_root"
