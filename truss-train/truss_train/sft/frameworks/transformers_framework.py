"""Transformers + PEFT framework for SFT."""

from truss_train.definitions import AutoSFT, CacheConfig, CheckpointingConfig, Runtime
from truss_train.sft.frameworks import FrameworkOutput, get_environment_variables


class TransformersFramework:
    name = "transformers"

    def generate(self, sft_config: AutoSFT) -> FrameworkOutput:
        return FrameworkOutput(
            runtime=Runtime(
                start_commands=[
                    "/bin/sh -c 'pip install -q transformers datasets peft accelerate && python train.py'"
                ],
                environment_variables=get_environment_variables(sft_config),
                cache_config=CacheConfig(enabled=True),
                checkpointing_config=CheckpointingConfig(enabled=True),
            ),
            train_script=_TRAIN_SCRIPT,
            base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime",
        )


_TRAIN_SCRIPT = '''#!/usr/bin/env python3
"""SFT training - transformers + PEFT."""
import json
import os
from pathlib import Path

def main():
    cfg = json.loads((Path(__file__).parent / "sft_config.json").read_text())
    from data_loader import UniversalLLMLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType

    model_id = cfg["model"]
    dataset_id = cfg["dataset"]
    num_epochs = cfg["num_epochs"]
    lr = cfg.get("learning_rate") or 2e-5
    optimizer = cfg.get("optimizer") or "adamw_torch"
    lr_scheduler = cfg.get("lr_scheduler") or "cosine"
    max_samples = cfg.get("max_samples")
    split = cfg.get("split")

    print(f"[*] Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05)
    model = get_peft_model(model, peft_config)

    print(f"[*] Loading dataset: {dataset_id}")
    loader = UniversalLLMLoader(model_name_or_path=model_id)
    ds = loader.load(dataset_id, split=split, max_samples=max_samples)
    if isinstance(ds, dict):
        ds = ds["train"] if "train" in ds else list(ds.values())[0]

    def format_example(example):
        messages = example["messages"]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = "\\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return {"text": text}

    ds = ds.map(format_example, remove_columns=ds.column_names)

    def tokenize_fn(examples):
        result = tokenizer(examples["text"], truncation=True, max_length=2048, padding="max_length")
        result["labels"] = [
            [tid if m else -100 for tid, m in zip(ids, mask)]
            for ids, mask in zip(result["input_ids"], result["attention_mask"])
        ]
        return result

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/output")
    os.makedirs(output_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            learning_rate=lr,
            optim=optimizer,
            lr_scheduler_type=lr_scheduler,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
        ),
        train_dataset=tokenized,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[*] Saved to {output_dir}")

if __name__ == "__main__":
    main()
'''
