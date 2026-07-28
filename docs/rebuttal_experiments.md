# Rebuttal experiment workflow

All annotation inputs in this workflow come from `editsleuth_data/release`.
The `external/editsleuth` tree is not used.

## 1. Install server dependencies

```bash
uv sync
```

All server-side Python commands below use the locked uv environment. Conda is
used only for local inspection on the macOS development machine. The server
must also have the AWS CLI:

```bash
aws --version
```

## 2. Download the EditSleuth release

Pass the public Google Drive file ID; do not commit the ID or a generated
download URL into the repository.

```bash
uv run python scripts/setup_editsleuth_release.py \
  --file-id GOOGLE_DRIVE_FILE_ID \
  --archive /srv/vanloc/editsleuth_data.tar.gz \
  --output-dir /srv/vanloc/editsleuth_data/release
```

The extractor refuses to merge into a non-empty release directory.

## 3. Download Pico-Banana

This follows Apple's official alternative to Flickr: source images come from
the two packed Open Images S3 archives using unsigned AWS CLI requests. Edited
images and manifests come from Apple's CDN.

```bash
uv run python scripts/download_pico_banana.py \
  --root /srv/vanloc/data/pico-banana-400k \
  --workers 16
```

Expected paths include:

```text
/data/pico-banana-400k/openimages/train_0/<image-id>.jpg
/data/pico-banana-400k/openimages/train_1/<image-id>.jpg
/data/pico-banana-400k/images/positive-edit/<edit-id>.png
```

The downloader is resumable at the file level. Use `--keep-archives` if the
Open Images tarballs should remain after successful extraction.

## 4. Download MagicBrush dev

The instruction-masked models are evaluated on the 528-row MagicBrush dev
split. A newly allocated server must materialize these images before launching
the experiment:

```bash
uv run python scripts/prepare_magicbrush_dev.py \
  --root /srv/vanloc/data/magicbrush \
  --annotations editsleuth_data/release/magicbrush_dev_annotations.parquet
```

This downloads `osunlp/MagicBrush` through Hugging Face and writes the exact
paths referenced by the release:

```text
/srv/vanloc/data/magicbrush/dev/magicbrush_dev_<id>_t<turn>__real.png
/srv/vanloc/data/magicbrush/dev/magicbrush_dev_<id>_t<turn>__edited.png
/srv/vanloc/data/magicbrush/dev/magicbrush_dev_<id>_t<turn>__mask.png
```

The command finishes by checking every `real_path` and `edited_path` in the
release Parquet. It is resumable because existing non-empty PNGs are retained.

## 5. Run the instruction-masked pilot

The chain and label-only runs train concurrently on GPUs 0 and 1. Evaluation
runs on GPU 2 after both adapters finish:

```bash
bash scripts/run_instruction_masked_ablation.sh \
  /srv/vanloc/data/pico-banana-400k \
  /srv/vanloc/data/magicbrush \
  outputs/rebuttal/instruction_masked
```

Both variants receive the original and edited images but not the instruction.
For chain training, Step 1 is replaced by a fixed withheld-instruction marker;
otherwise the loss would require reproducing text deliberately absent from the
input. Steps 2–6 are unchanged.

For a quick pipeline check, invoke each training command directly with:

```text
samples_per_category_per_bin=1
```

## 6. Audit 200 reasoning traces

Create a deterministic 67/67/66 easy/medium/hard sample:

```bash
uv run python scripts/audit_reasoning_traces.py sample \
  --input editsleuth_data/release/pico_banana_annotations.parquet \
  --output outputs/rebuttal/trace_audit_200.csv \
  --n 200 \
  --seed 2026
```

For each step, an annotator fills:

- `step_N_correct`: `yes`, `no`, or `unclear`
- `step_N_error_type`: a consistent error taxonomy
- `step_N_notes`: concise evidence for the judgment

After all judgments are complete:

```bash
uv run python scripts/audit_reasoning_traces.py report \
  --input outputs/rebuttal/trace_audit_200.csv \
  --output outputs/rebuttal/trace_audit_200_report.json
```

The report includes overall and per-difficulty error rates for every step,
complete-chain accuracy, unclear counts, and the first failing step per trace.
