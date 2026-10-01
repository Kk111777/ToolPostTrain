#!/usr/bin/env python3
"""Minimal PEFT LoRA SFT runner for the prepared ToolRL format dataset.

This is an independent SFT entry point. It does not import or modify the
GRPO/GDPO trainer or the reward manager.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL = (
    "/root/autodl-tmp/ProjectB/hf_cache/models--Qwen--"
    "Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
DEFAULT_DATA = (
    "/root/autodl-tmp/ProjectB/repo/dataset/format_sft_800.jsonl"
)
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/ProjectB/checkpoints/format_sft_lora"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def input_ids(encoded: Any) -> list[int]:
    try:
        values = encoded["input_ids"]
    except (KeyError, TypeError, IndexError):
        values = encoded
    if values and isinstance(values[0], list):
        values = values[0]
    return list(values)


class FormatSFTDataset(Dataset):
    """Tokenize chat messages and mask the system/user prefix in labels."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        max_seq_length: int,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        data_path = Path(path)
        if not data_path.is_file():
            raise FileNotFoundError(data_path)

        with data_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                messages = record.get("messages")
                if not isinstance(messages, list) or len(messages) < 3:
                    raise ValueError(
                        f"{data_path}:{line_number} must contain "
                        "system, user, assistant messages"
                    )
                if messages[-1].get("role") != "assistant":
                    raise ValueError(
                        f"{data_path}:{line_number} final message is not assistant"
                    )

                prompt_messages = messages[:-1]
                full_ids = input_ids(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                )
                prompt_ids = input_ids(
                    tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    raise ValueError(
                        f"{data_path}:{line_number} chat-template prefix mismatch"
                    )
                if len(full_ids) > max_seq_length:
                    raise ValueError(
                        f"{data_path}:{line_number} has {len(full_ids)} tokens, "
                        f"above max_seq_length={max_seq_length}"
                    )
                if len(full_ids) == len(prompt_ids):
                    raise ValueError(
                        f"{data_path}:{line_number} has no assistant loss tokens"
                    )

                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
                attention_mask = [1] * len(full_ids)
                self.examples.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                )

        if not self.examples:
            raise ValueError(f"No examples found in {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CausalLMCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        input_batch = []
        attention_batch = []
        label_batch = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_batch.append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )
            attention_batch.append(feature["attention_mask"] + [0] * padding)
            label_batch.append(feature["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
            "labels": torch.tensor(label_batch, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This SFT runner requires a CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The current CUDA device does not support BF16.")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = FormatSFTDataset(
        path=args.data_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    if trainer.is_world_process_zero():
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "format_sft_run_config.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(vars(args), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    print(f"adapter_output={Path(args.output_dir).resolve()}")
    print("FORMAT_SFT_TRAIN_PASS")


if __name__ == "__main__":
    main()
