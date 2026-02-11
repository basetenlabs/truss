"""DeepSpeed ZeRO-3 + transformers for multinode distributed SFT."""

import json

from truss_train.definitions import AutoSFT, CacheConfig, CheckpointingConfig, Runtime
from truss_train.sft.frameworks import FrameworkOutput, get_environment_variables

GPUS_PER_NODE = 8


class DeepSpeedFramework:
    name = "deepspeed"

    def generate(self, sft_config: AutoSFT) -> FrameworkOutput:
        node_count = sft_config.node_count or 1
        ds_config = dict(_DS_CONFIG)
        if node_count > 1:
            ds_config["checkpoint"] = {"use_node_local_storage": True}

        # torchrun for multinode; platform sets NODE_RANK, MASTER_ADDR, MASTER_PORT
        rdzv_timeout = "--rdzv-timeout=3600 " if node_count > 1 else ""
        launch_cmd = (
            f"torchrun --nproc_per_node={GPUS_PER_NODE} --nnodes={node_count} "
            f"{rdzv_timeout}"
            "--node_rank=${NODE_RANK:-0} "
            "--master_addr=${MASTER_ADDR:-localhost} "
            "--master_port=${MASTER_PORT:-29500} "
            "train.py"
        )
        train_script = _TRAIN_SCRIPT.replace(
            "_DS_CONFIG_PLACEHOLDER_", repr(json.dumps(ds_config, indent=2))
        )
        return FrameworkOutput(
            runtime=Runtime(
                start_commands=[
                    f"/bin/sh -c 'pip install -q transformers datasets peft accelerate deepspeed && {launch_cmd}'"
                ],
                environment_variables=get_environment_variables(sft_config),
                cache_config=CacheConfig(enabled=True),
                checkpointing_config=CheckpointingConfig(enabled=True),
            ),
            train_script=train_script,
            base_image="baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3",
        )


_DS_CONFIG = {
    "bf16": {"enabled": "auto"},
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto"},
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {"warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto"},
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

_TRAIN_SCRIPT = '''#!/usr/bin/env python3
"""SFT training - DeepSpeed ZeRO-3 + transformers (multinode)."""
import json
import os
import socket
from pathlib import Path

def main():
    # Debug: print multinode environment info
    print("=" * 60, flush=True)
    print("[DEBUG] Multinode environment variables:", flush=True)
    print(f"  MASTER_ADDR = {os.environ.get('MASTER_ADDR', '(not set)')}", flush=True)
    print(f"  MASTER_PORT = {os.environ.get('MASTER_PORT', '(not set)')}", flush=True)
    print(f"  NODE_RANK = {os.environ.get('NODE_RANK', '(not set)')}", flush=True)
    print(f"  WORLD_SIZE = {os.environ.get('WORLD_SIZE', '(not set)')}", flush=True)
    print(f"  RANK = {os.environ.get('RANK', '(not set)')}", flush=True)
    print(f"  LOCAL_RANK = {os.environ.get('LOCAL_RANK', '(not set)')}", flush=True)
    print(f"  Hostname = {socket.gethostname()}", flush=True)
    print("=" * 60, flush=True)
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

    # Write DeepSpeed ZeRO-3 config
    ds_config_path = Path(__file__).parent / "ds_config.json"
    ds_config_path.write_text(_DS_CONFIG_PLACEHOLDER_)

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

    node_count = cfg.get("node_count", 1)
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
            deepspeed=str(ds_config_path),
            gradient_checkpointing=True,
            save_on_each_node=node_count > 1,
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
