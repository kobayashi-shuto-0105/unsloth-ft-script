#!/usr/bin/env python3
"""Text-only SFT for Qwen3.5-35B-A3B with Unsloth.

This script is designed for the current local workflow:

- model: unsloth/Qwen3.5-35B-A3B, usually loaded from a local HF snapshot path
- dataset: Luoberta/cve_train_v1.1, usually loaded from a local datasets.save_to_disk path
- training: bf16 LoRA, text-only SFT over the dataset's messages field

It deliberately converts the dataset to a plain `text` column before constructing
SFTTrainer. This avoids TRL treating Qwen3VLProcessor as a VLM processor and
requiring `images` in each sample.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Unsloth should be imported before trl / transformers / peft.
import unsloth  # noqa: F401
from unsloth import FastModel, FastVisionModel

import torch
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-35B-A3B text-only LoRA SFT")

    parser.add_argument("--model-path", required=True, help="Local model snapshot path or HF model id if --allow-remote is set.")
    parser.add_argument("--dataset-path", required=True, help="Local datasets.save_to_disk directory or dataset id if --allow-remote is set.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-remote", action="store_true", help="Allow loading remote HF ids instead of strict local paths.")

    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=None)

    parser.add_argument("--lora-r", "--lora-rank", dest="lora_r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--modules-to-save", action="store_true", help="Also save lm_head and embed_tokens in the adapter.")

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit-samples", type=int, default=512, help="0 means use all filtered samples.")
    parser.add_argument("--max-turns", type=int, default=80, help="0 disables turn-count filtering.")
    parser.add_argument("--include-unresolved", action="store_true")
    parser.add_argument("--packing", action="store_true")

    parser.add_argument("--use-uma-eager-safe-open", action="store_true", help="Enable experimental GB10/DGX Spark UMA safetensors loading workaround.")
    parser.add_argument("--no-uma-eager-safe-open", action="store_true", help="Explicitly disable the UMA workaround.")

    return parser.parse_args()


def enable_offline_if_needed(args: argparse.Namespace) -> None:
    if not args.allow_remote:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"


def sft_fields() -> dict[str, Any]:
    return getattr(SFTConfig, "__dataclass_fields__", {}) or {}


def set_if_supported(config: dict[str, Any], key: str, value: Any) -> None:
    if key in sft_fields():
        config[key] = value


def set_length_arg(config: dict[str, Any], max_seq_length: int) -> None:
    fields = sft_fields()
    if "max_length" in fields:
        config["max_length"] = max_seq_length
    elif "max_seq_length" in fields:
        config["max_seq_length"] = max_seq_length
    else:
        raise RuntimeError("SFTConfig supports neither max_length nor max_seq_length")


def make_trainer_kwargs(model: Any, processing_class: Any, dataset: Any, args: SFTConfig, data_collator: Any) -> dict[str, Any]:
    signature = inspect.signature(SFTTrainer.__init__).parameters
    kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": args,
        "data_collator": data_collator,
    }
    if "processing_class" in signature:
        kwargs["processing_class"] = processing_class
    elif "tokenizer" in signature:
        kwargs["tokenizer"] = processing_class
    return kwargs


def assert_model_weights(model_path: str) -> None:
    path = Path(model_path)
    if not path.exists():
        if "/" in model_path:
            # Could be a remote HF id when --allow-remote is set.
            return
        raise FileNotFoundError(f"Model path not found: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Model path is not a directory: {path}")

    hits = []
    for pattern in [
        "model.safetensors",
        "model.safetensors.index.json",
        "model.safetensors-*.safetensors",
        "*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "*.bin",
    ]:
        hits.extend(path.glob(pattern))

    hits = [item for item in hits if item.exists()]
    if not hits:
        top = sorted(item.name for item in path.iterdir())[:80]
        raise FileNotFoundError(
            f"No model weight files found in {path}. Top-level files: {top}. "
            "Pass the real HF snapshot path that contains model.safetensors shards."
        )

    print("weight files detected:")
    for item in sorted(hits)[:20]:
        try:
            real = item.resolve()
            size_gib = real.stat().st_size / 1024**3
            print(f"  {item.name} {size_gib:.2f} GiB")
        except Exception:
            print(f"  {item.name}")


def text_tokenizer(processing_class: Any) -> Any:
    return getattr(processing_class, "tokenizer", processing_class)


def token_exists(tokenizer: Any, token: str | None) -> bool:
    if not token:
        return False
    try:
        return token in tokenizer.get_vocab()
    except Exception:
        pass
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        return token_id is not None and token_id != unknown_id
    except Exception:
        return False


def fix_eos_pad(processing_class: Any) -> str:
    tokenizer = text_tokenizer(processing_class)
    print("=== tokenizer / processor ===")
    print("processing_class:", type(processing_class).__name__)
    print("text_tokenizer:", type(tokenizer).__name__)
    print("before processor eos_token:", getattr(processing_class, "eos_token", None))
    print("before tokenizer eos_token:", getattr(tokenizer, "eos_token", None))
    print("before tokenizer pad_token:", getattr(tokenizer, "pad_token", None))

    candidates = [
        getattr(tokenizer, "eos_token", None),
        getattr(processing_class, "eos_token", None),
        "<|im_end|>",
        "<|endoftext|>",
        "</s>",
    ]

    eos = None
    for candidate in candidates:
        if token_exists(tokenizer, candidate):
            eos = candidate
            break
    if eos is None:
        raise RuntimeError(f"Could not find a valid EOS token. Candidates: {candidates}")

    tokenizer.eos_token = eos
    pad_token = getattr(tokenizer, "pad_token", None)
    if pad_token is None or not token_exists(tokenizer, pad_token):
        tokenizer.pad_token = eos

    try:
        processing_class.eos_token = eos
    except Exception:
        pass

    print("fixed eos_token:", eos)
    print("fixed eos_token_id:", tokenizer.convert_tokens_to_ids(eos))
    print("fixed pad_token:", getattr(tokenizer, "pad_token", None))
    print("fixed pad_token_id:", getattr(tokenizer, "pad_token_id", None))
    return eos


def fallback_qwen_chat_template(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                else:
                    text_parts.append(str(item))
            content = "\n".join(text_parts)
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def load_training_dataset(args: argparse.Namespace, tokenizer: Any) -> Any:
    dataset_path = Path(args.dataset_path)
    if dataset_path.exists():
        print(f"Loading local dataset: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))
    elif args.allow_remote:
        print(f"Loading remote dataset: {args.dataset_path}")
        dataset = load_dataset(args.dataset_path, split="train")
    else:
        raise FileNotFoundError(f"Local dataset not found: {dataset_path}")

    print("raw dataset:", dataset)
    print("raw rows:", len(dataset))
    print("sample keys:", dataset[0].keys())
    print("sample turns:", len(dataset[0]["messages"]))

    if not args.include_unresolved:
        dataset = dataset.filter(lambda row: bool(row["is_resolved"]), num_proc=1)
        print("after resolved_only:", len(dataset))

    if args.max_turns and args.max_turns > 0:
        dataset = dataset.filter(lambda row: len(row["messages"]) <= args.max_turns, num_proc=1)
        print(f"after max_turns<={args.max_turns}:", len(dataset))

    dataset = dataset.shuffle(seed=args.seed)

    if args.limit_samples and args.limit_samples > 0:
        dataset = dataset.select(range(min(args.limit_samples, len(dataset))))
        print(f"after limit_samples={args.limit_samples}:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after filters")

    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(dataset) / effective_batch)
    print("=== estimated train size ===")
    print("train rows:", len(dataset))
    print("effective batch:", effective_batch)
    print("estimated optimizer steps / epoch:", steps_per_epoch)
    print("estimated total optimizer steps:", math.ceil(steps_per_epoch * args.epochs))
    if args.max_steps > 0:
        print("WARNING: max_steps > 0 overrides epochs in Trainer semantics")

    print("=== convert dataset to text-only ===")
    print("trainer tokenizer:", type(tokenizer).__name__)
    print("has apply_chat_template:", hasattr(tokenizer, "apply_chat_template"))

    def to_text(example: dict[str, Any]) -> dict[str, str]:
        messages = example["messages"]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception as exc:
                print("WARNING: apply_chat_template failed; using fallback:", repr(exc))
                text = fallback_qwen_chat_template(messages)
        else:
            text = fallback_qwen_chat_template(messages)
        return {"text": text}

    dataset = dataset.map(
        to_text,
        remove_columns=list(dataset.column_names),
        num_proc=1,
        desc="format text-only dataset",
    )
    print("text-only dataset:", dataset)
    print("sample text prefix:")
    print(dataset[0]["text"][:1000])
    return dataset


def main() -> None:
    args = parse_args()
    enable_offline_if_needed(args)

    print("=== config ===")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    print("=== env ===")
    for key in [
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "UNSLOTH_MOE_BACKEND",
        "HF_ENABLE_PARALLEL_LOADING",
        "HF_PARALLEL_LOADING_WORKERS",
    ]:
        print(f"{key}={os.environ.get(key)}")

    print("=== cuda ===")
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        print("total GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)

    print("=== load model ===")
    assert_model_weights(args.model_path)

    if args.use_uma_eager_safe_open and not args.no_uma_eager_safe_open:
        from uma_eager_safe_open import install_uma_eager_safe_open

        install_uma_eager_safe_open()

    model, processing_class = FastModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map={"": 0},
        trust_remote_code=True,
    )

    device_map = getattr(model, "hf_device_map", {}) or {}
    offloaded = {key: value for key, value in device_map.items() if str(value) in {"cpu", "disk"}}
    print("offloaded:", offloaded)
    if offloaded:
        raise RuntimeError(f"CPU/disk offload detected: {offloaded}")

    print("allocated GiB after load:", torch.cuda.memory_allocated() / 1024**3)
    print("reserved GiB after load:", torch.cuda.memory_reserved() / 1024**3)

    eos_token = fix_eos_pad(processing_class)
    tokenizer = text_tokenizer(processing_class)

    print("=== attach LoRA ===")
    peft_kwargs = dict(
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )
    if args.modules_to_save:
        peft_kwargs["modules_to_save"] = ["lm_head", "embed_tokens"]

    model = FastVisionModel.get_peft_model(model, **peft_kwargs)

    print("allocated GiB after LoRA:", torch.cuda.memory_allocated() / 1024**3)
    print("reserved GiB after LoRA:", torch.cuda.memory_reserved() / 1024**3)

    dataset = load_training_dataset(args, tokenizer)

    print("=== build SFTConfig ===")
    config = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="no" if args.save_steps is None else "steps",
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        dataset_num_proc=1,
        packing=args.packing,
        assistant_only_loss=False,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
    )
    if args.save_steps is not None:
        config["save_steps"] = args.save_steps
    if args.warmup_ratio is not None:
        set_if_supported(config, "warmup_ratio", args.warmup_ratio)
    else:
        set_if_supported(config, "warmup_steps", args.warmup_steps)

    set_if_supported(config, "eos_token", eos_token)
    set_if_supported(config, "dataset_text_field", "text")
    set_length_arg(config, args.max_seq_length)

    fields = sft_fields()
    config = {key: value for key, value in config.items() if key in fields}
    for key, value in config.items():
        print(f"  {key}: {value}")

    training_args = SFTConfig(**config)

    print("=== build text-only data collator ===")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("=== train ===")
    trainer = SFTTrainer(
        **make_trainer_kwargs(
            model=model,
            processing_class=tokenizer,
            dataset=dataset,
            args=training_args,
            data_collator=data_collator,
        )
    )
    trainer.train()

    print("=== save LoRA adapter ===")
    adapter_dir = Path(args.output_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("saved:", adapter_dir)
    print("done")


if __name__ == "__main__":
    main()
