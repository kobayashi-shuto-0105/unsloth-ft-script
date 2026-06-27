#!/usr/bin/env python3
"""Download/materialize model and dataset assets for local/offline training.

Example:

    python download_assets.py \
      --model-id unsloth/Qwen3.5-35B-A3B \
      --model-dir /workspace/models/Qwen3.5-35B-A3B \
      --dataset-id Luoberta/cve_train_v1.1 \
      --dataset-dir /workspace/datasets/Luoberta_cve_train_v1.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="unsloth/Qwen3.5-35B-A3B")
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-35B-A3B")
    parser.add_argument("--dataset-id", default="Luoberta/cve_train_v1.1")
    parser.add_argument("--dataset-dir", default="/workspace/datasets/Luoberta_cve_train_v1.1")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    dataset_dir = Path(args.dataset_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    print("=== download model snapshot ===")
    model_path = snapshot_download(
        repo_id=args.model_id,
        repo_type="model",
        local_dir=str(model_dir),
        revision=args.revision,
    )
    print("model_path:", model_path)

    print("=== download and save dataset ===")
    dataset = load_dataset(args.dataset_id, split="train")
    dataset.save_to_disk(str(dataset_dir))
    print(dataset)
    print("dataset_dir:", dataset_dir)

    print("done")


if __name__ == "__main__":
    main()
