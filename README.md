# unsloth-ft-script

Unsloth fine-tuning helper scripts and notes.

## Contents

- [`unsloth-Qwen3.5-35B-A3B/`](./unsloth-Qwen3.5-35B-A3B/) — Qwen3.5-35B-A3B bf16 LoRA SFT scripts for local model/dataset use, with optional DGX Spark / GB10 UMA load-peak workaround.

## Current target

The current scripts are focused on:

- Model: `unsloth/Qwen3.5-35B-A3B`
- Dataset: `Luoberta/cve_train_v1.1`
- Environment: Unsloth Studio Python environment on NVIDIA GB10 / DGX Spark-like unified-memory machines
- Training style: text-only SFT over CVE agent traces

See the folder README for exact setup, quick start, options, and troubleshooting.
