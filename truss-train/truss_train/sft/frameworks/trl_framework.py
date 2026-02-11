"""TRL (SFTTrainer) framework for SFT."""

from truss_train.definitions import AutoSFT, CacheConfig, CheckpointingConfig, Runtime
from truss_train.sft.frameworks import FrameworkOutput, get_environment_variables


class TRLFramework:
    name = "trl"

    def generate(self, sft_config: AutoSFT) -> FrameworkOutput:
        return FrameworkOutput(
            runtime=Runtime(
                start_commands=[
                    "/bin/sh -c 'pip install -q transformers datasets peft accelerate trl && python train.py'"
                ],
                environment_variables=get_environment_variables(sft_config),
                cache_config=CacheConfig(enabled=True),
                checkpointing_config=CheckpointingConfig(enabled=True),
            ),
            train_script=_TRAIN_SCRIPT,
            base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime",
        )


_TRAIN_SCRIPT = '''#!/usr/bin/env python3
"""SFT training - TRL SFTTrainer."""
import json
import os
from pathlib import Path

def main():
    cfg = json.loads((Path(__file__).parent / "sft_config.json").read_text())
    from data_loader import UniversalLLMLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer
    from peft import LoraConfig, TaskType

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

    print(f"[*] Loading dataset: {dataset_id}")
    loader = UniversalLLMLoader(model_name_or_path=model_id)
    ds = loader.load(dataset_id, split=split, max_samples=max_samples)
    if isinstance(ds, dict):
        ds = ds["train"] if "train" in ds else list(ds.values())[0]

    def format_to_text(example):
        messages = example["messages"]
        return {"text": "\\n".join(f"{m['role']}: {m['content']}" for m in messages)}

    ds = ds.map(format_to_text, remove_columns=ds.column_names)

    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/output")
    os.makedirs(output_dir, exist_ok=True)

    trainer = SFTTrainer(
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
        train_dataset=ds,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=2048,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[*] Saved to {output_dir}")

if __name__ == "__main__":
    main()
'''
