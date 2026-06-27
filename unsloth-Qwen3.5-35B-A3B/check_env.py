#!/usr/bin/env python3
"""Print environment details for the Qwen3.5 Unsloth workflow."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="Optional local model path to inspect with AutoConfig/AutoTokenizer.")
    parser.add_argument("--dataset-path", default=None, help="Optional local dataset save_to_disk path to inspect.")
    args = parser.parse_args()

    import torch
    import transformers
    import unsloth

    print("python:", sys.executable)
    print("unsloth:", getattr(unsloth, "__version__", "unknown"))
    try:
        import unsloth_zoo

        print("unsloth_zoo:", getattr(unsloth_zoo, "__version__", "unknown"))
    except Exception as exc:
        print("unsloth_zoo ERROR:", repr(exc))

    print("transformers:", transformers.__version__)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        print("arch list:", torch.cuda.get_arch_list())
        print("total memory GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)

    for package in ["accelerate", "bitsandbytes", "triton", "huggingface_hub", "tokenizers", "datasets", "trl"]:
        try:
            mod = __import__(package)
            print(f"{package}:", getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            print(f"{package} ERROR:", repr(exc))

    if args.model_path:
        print("=== model check ===")
        from transformers import AutoConfig, AutoTokenizer

        cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        print("model_type:", cfg.model_type)
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        print("tokenizer:", type(tok).__name__, len(tok))

    if args.dataset_path:
        print("=== dataset check ===")
        from datasets import load_from_disk

        ds = load_from_disk(args.dataset_path)
        print(ds)
        print("rows:", len(ds))
        print("keys:", ds[0].keys())
        if "messages" in ds[0]:
            print("sample turns:", len(ds[0]["messages"]))


if __name__ == "__main__":
    main()
