# Remaining rebuttal experiments

## One-command server preparation

The launcher uses the instruction-visible label-only model for saliency. If its
adapter is absent, it first retrains that control on GPU 0. It then runs
saliency on GPU 0 and a matched free-form VLM baseline on GPU 1 concurrently:

```bash
bash scripts/run_remaining_rebuttal_experiments.sh \
  /srv/vanloc/editsleuth_data/release \
  /srv/vanloc/data/pico-banana-400k \
  /srv/vanloc/data/magicbrush \
  /srv/vanloc/data/outputs/rebuttal/remaining
```

Run this inside `tmux`. It produces:

```text
remaining/
├── visible_label_only/
├── label_only_saliency/
├── freeform_vlm_chains_200.parquet
├── human_difficulty_audit_100/
└── paired_chain_audit_200/
```

## Saliency protocol

`generate_label_saliency.py` uses input-gradient × input magnitude for the
log-probability of the complete structured label prediction. It exports
separate heatmaps and overlays for the original and edited images. The default
50 examples are balanced across category/difficulty cells.

These maps are a qualitative explanation channel. They are not localization
ground truth and do not establish that the highlighted pixels are causally
responsible for the prediction.

## Human difficulty protocol

The 100-triplet bundle is balanced across computed difficulty and category.
`annotations.csv` hides the computed score and bin; `answer_key.csv` must remain
closed until both annotators finish independently.

After annotation:

```bash
uv run python scripts/audit_human_difficulty.py report \
  --annotations human_difficulty_audit_100/annotations.csv \
  --answer-key human_difficulty_audit_100/answer_key.csv \
  --output human_difficulty_audit_100/report.json
```

The report includes per-annotator and consensus agreement with the computed
bin, Spearman correlation with the raw score, exact agreement, and quadratic
weighted kappa.

## Matched free-form comparison and 200-trace audit

The baseline uses unfine-tuned `Qwen/Qwen2-VL-2B-Instruct` on the same 200
difficulty-stratified image pairs. It receives the original image, edited
image, and instruction, then generates six concise steps without invented
numerical measurements. Generation is greedy and resumable through the
`.progress.jsonl` file.

`paired_chain_audit_200` randomizes whether the computed or free-form trace is
shown as A or B for each triplet. Annotators judge both systems with the same
per-step rubric.

After both annotators finish:

```bash
uv run python scripts/prepare_paired_chain_audit.py report \
  --annotations paired_chain_audit_200/annotations.csv \
  --answer-key paired_chain_audit_200/answer_key.csv \
  --output paired_chain_audit_200/report.json
```

The report provides per-step and per-difficulty error rates, first-error
locations, complete-chain accuracy, formatting compliance, and paired exact
McNemar tests for computed versus free-form chains.
