# Qwen3.5-35B-A3B Unsloth fine-tuning scripts

Scripts and operational notes for fine-tuning `unsloth/Qwen3.5-35B-A3B` with Unsloth on a local machine/container.

This folder was written for the current GB10 / DGX Spark-like workflow where:

- Unsloth Studio provides the Python environment at `/root/.unsloth/studio/unsloth_studio/bin/python`
- The model and dataset are already cached locally
- The target dataset is `Luoberta/cve_train_v1.1`
- The run is bf16 LoRA, not 4-bit QLoRA

## Files

| File | Purpose |
|---|---|
| `train_qwen35_cve_textonly.py` | Main training script. Loads Qwen3.5-35B-A3B, attaches LoRA, converts `messages` to text, and runs SFT. |
| `uma_eager_safe_open.py` | Optional DGX Spark / GB10 unified-memory safetensors load-peak workaround. |
| `check_env.py` | Prints key Python/package/GPU/model/dataset information. |
| `download_assets.py` | Downloads/materializes the model snapshot and saves the dataset to disk. |

## Why this is text-only

`Qwen3.5-35B-A3B` loads through a vision-language processor (`Qwen3VLProcessor`) in this Unsloth/Transformers stack, but `Luoberta/cve_train_v1.1` is a text conversation dataset with columns:

```text
['task_id', 'is_resolved', 'messages']
```

Passing the VLM processor directly to `SFTTrainer` makes TRL expect an `images` field. This script instead converts `messages` into a `text` column and passes the internal text tokenizer to a language-modeling data collator.

## Quick start

### 1. Set paths

```bash
export STUDIO_PY=/root/.unsloth/studio/unsloth_studio/bin/python
export HF_HOME=/workspace/.cache/huggingface
```

Find the local model snapshot:

```bash
export SNAPSHOT="$(find /root/.cache/huggingface/hub /workspace/.cache/huggingface/hub \
  -type d -path '*models--unsloth--Qwen3.5-35B-A3B/snapshots/*' 2>/dev/null | head -1)"

echo "$SNAPSHOT"
test -n "$SNAPSHOT"
```

Expected local dataset path:

```bash
export DATASET_PATH=/workspace/datasets/Luoberta_cve_train_v1.1
```

### 2. Optional: drop Linux page cache before a run

Useful on unified-memory systems when loading 35B bf16 weights from safetensors:

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

### 3. Run a 512-sample smoke test

```bash
docker exec -i unsloth bash <<'BASH'
set -euxo pipefail
STUDIO_PY=/root/.unsloth/studio/unsloth_studio/bin/python

SNAPSHOT="$(find /root/.cache/huggingface/hub /workspace/.cache/huggingface/hub \
  -type d -path '*models--unsloth--Qwen3.5-35B-A3B/snapshots/*' 2>/dev/null | head -1)"

test -n "$SNAPSHOT"

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_ENABLE_PARALLEL_LOADING=false
export HF_PARALLEL_LOADING_WORKERS=1
export UNSLOTH_MOE_BACKEND=grouped_mm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace:${PYTHONPATH:-}

"$STUDIO_PY" /workspace/unsloth-Qwen3.5-35B-A3B/train_qwen35_cve_textonly.py \
  --model-path "$SNAPSHOT" \
  --dataset-path /workspace/datasets/Luoberta_cve_train_v1.1 \
  --max-seq-length 1024 \
  --epochs 1 \
  --limit-samples 512 \
  --max-turns 80 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-r 8 \
  --lora-alpha 8 \
  --learning-rate 2e-5 \
  --warmup-steps 10 \
  --use-uma-eager-safe-open \
  --output-dir /workspace/outputs/qwen35-cve-local-ctx1024-ep1-smoke
BASH
```

### 4. Run filtered one-epoch training

After the smoke test works, remove the sample limit:

```bash
--limit-samples 0
```

This trains on:

```text
is_resolved == true
messages length <= --max-turns
```

## Common options

| Option | Default | Meaning |
|---|---:|---|
| `--model-path` | required | Local HF snapshot directory containing `model.safetensors-*` shards. |
| `--dataset-path` | required | Local `datasets.save_to_disk()` directory. |
| `--max-seq-length` | `1024` | SFT truncation length. Start at 1024, then try 2048. |
| `--epochs` | `1.0` | Number of epochs. Ignored if `--max-steps > 0`. |
| `--max-steps` | `-1` | Positive values override epoch-based training. |
| `--limit-samples` | `512` | Smoke-test limit. Use `0` for all filtered samples. |
| `--max-turns` | `80` | Drop traces with more than this many messages. Use `0` to disable. |
| `--include-unresolved` | off | Include `is_resolved=false` traces. Usually keep off for first runs. |
| `--lora-r` / `--lora-rank` | `8` | LoRA rank. Try `4` if memory is tight, `16` after stable. |
| `--lora-alpha` | `8` | LoRA alpha. Usually match rank for this smoke workflow. |
| `--use-uma-eager-safe-open` | off | Enable experimental GB10/DGX Spark UMA safetensors peak workaround. |
| `--no-uma-eager-safe-open` | off | Explicitly disable the workaround. |
| `--packing` | off | Enable TRL packing. Leave off for debugging. |

## Expected milestones

During a successful run you should see:

```text
Installed UMA eager safe_open patch ...     # only when --use-uma-eager-safe-open
Loading weights: 100%
offloaded: {}
allocated GiB after load: ~65
allocated GiB after LoRA: ~67
text-only dataset: Dataset({ features: ['text'], ... })
=== train ===
loss: ...
```

## Troubleshooting

### `Error no file named model.safetensors...`

The directory passed as `--model-path` does not contain real weight shards. Use the HF cache snapshot path, not the incomplete `/workspace/models/Qwen3.5-35B-A3B` directory.

### `Some parameters are on the meta device because they were offloaded to the cpu`

Do not use `device_map='sequential'` for this training path. This script uses `device_map={'': 0}` and stops if CPU/disk offload is detected.

### `eos_token ('<EOS_TOKEN>') is not found`

This is caused by TRL defaulting to a placeholder EOS token with `Qwen3VLProcessor`. The script detects the real Qwen EOS token, usually `<|im_end|>`, and passes it to `SFTConfig`.

### `KeyError: 'images'`

This happens when TRL treats the run as VLM training. This script avoids that by converting the dataset to a `text` column and using the internal text tokenizer rather than the full VLM processor.

### `assistant_only_loss is not yet supported for vision-language models`

This script is intentionally text-only and defaults to full-token language-modeling loss after converting messages to text. Assistant-only masking can be added later by building labels manually.

## References

- Unsloth Qwen3.5 fine-tuning docs: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- TRL SFTTrainer docs: https://huggingface.co/docs/trl/en/sft_trainer
- Transformers big model loading docs: https://huggingface.co/docs/transformers/main/en/big_models
- Safetensors docs: https://huggingface.co/docs/safetensors/index
