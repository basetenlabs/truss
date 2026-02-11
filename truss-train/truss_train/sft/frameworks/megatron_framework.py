"""MS-Swift/Megatron framework for multinode distributed SFT."""

from truss_train.definitions import AutoSFT, CacheConfig, CheckpointingConfig, Runtime
from truss_train.sft.frameworks import FrameworkOutput, get_environment_variables


class MegatronFramework:
    name = "megatron"

    def generate(self, sft_config: AutoSFT) -> FrameworkOutput:
        model_params = sft_config.model_params_b or 70
        num_epochs = sft_config.num_epochs or 1
        train_iters = num_epochs * 100
        lr = sft_config.learning_rate or 1e-4
        min_lr = lr / 10
        
        # Parallelism settings based on model size
        if model_params > 100:
            # Large MoE models like Qwen3-235B
            tp_size = 4
            ep_size = 16
            moe_flags = """    --moe_permute_fusion true \\
    --moe_grouped_gemm true \\
    --moe_shared_expert_overlap true \\
    --moe_aux_loss_coeff 1e-3 \\"""
            max_length = 16384
        elif model_params > 30:
            # Large dense models like Llama-70B
            tp_size = 8
            ep_size = 1
            moe_flags = ""
            max_length = 16384
        else:
            # Smaller models
            tp_size = 4
            ep_size = 1
            moe_flags = ""
            max_length = 8192
        
        # Generate run.sh script content (matches user's working script exactly)
        if model_params > 100:
            # Large MoE model script - exact match to working run.sh
            run_script = f"""#!/bin/bash
export HF_HOME=$BT_RW_CACHE_DIR/huggingface

checkpoint_dir="$BT_CHECKPOINT_DIR/output"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NPROC_PER_NODE=$BT_NUM_GPUS NNODES=$BT_GROUP_SIZE NODE_RANK=$BT_NODE_RANK MASTER_ADDR=$BT_LEADER_ADDR megatron sft \\
    --model {sft_config.model} \\
    --save $checkpoint_dir \\
    --dataset '{sft_config.dataset}' \\
    --load_safetensors true \\
    --save_safetensors true \\
    --train_type lora \\
    --lora_rank 64 \\
    --lora_alpha 128 \\
    --target_modules all-linear \\
    --no_initialization false \\
    --split_dataset_ratio 0.01 \\
    --tensor_model_parallel_size {tp_size} \\
    --expert_model_parallel_size {ep_size} \\
    --recompute_num_layers 2 \\
    --moe_permute_fusion true \\
    --moe_grouped_gemm true \\
    --moe_shared_expert_overlap true \\
    --moe_aux_loss_coeff 1e-3 \\
    --micro_batch_size 1 \\
    --global_batch_size 4 \\
    --packing true \\
    --recompute_granularity full \\
    --recompute_method uniform \\
    --train_iters {train_iters} \\
    --eval_iters 40 \\
    --finetune true \\
    --cross_entropy_loss_fusion true \\
    --lr {lr} \\
    --lr_warmup_fraction 0.05 \\
    --min_lr {min_lr} \\
    --eval_interval 40 \\
    --max_length {max_length} \\
    --num_workers 8 \\
    --dataset_num_proc 8 \\
    --no_save_optim true \\
    --no_save_rng true \\
    --sequence_parallel true \\
    --attention_backend flash \\
    --optimizer_cpu_offload true \\
    --use_precision_aware_optimizer true \\
    --merge_lora false \\
    --use_hf 1
"""
        else:
            # Dense model script
            run_script = f"""#!/bin/bash
export HF_HOME=$BT_RW_CACHE_DIR/huggingface

checkpoint_dir="$BT_CHECKPOINT_DIR/output"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NPROC_PER_NODE=$BT_NUM_GPUS NNODES=$BT_GROUP_SIZE NODE_RANK=$BT_NODE_RANK MASTER_ADDR=$BT_LEADER_ADDR megatron sft \\
    --model {sft_config.model} \\
    --save $checkpoint_dir \\
    --dataset '{sft_config.dataset}' \\
    --load_safetensors true \\
    --save_safetensors true \\
    --train_type lora \\
    --lora_rank 64 \\
    --lora_alpha 128 \\
    --target_modules all-linear \\
    --no_initialization false \\
    --split_dataset_ratio 0.01 \\
    --tensor_model_parallel_size {tp_size} \\
    --recompute_num_layers 2 \\
    --micro_batch_size 1 \\
    --global_batch_size 4 \\
    --packing true \\
    --recompute_granularity full \\
    --recompute_method uniform \\
    --train_iters {train_iters} \\
    --eval_iters 40 \\
    --finetune true \\
    --cross_entropy_loss_fusion true \\
    --lr {lr} \\
    --lr_warmup_fraction 0.05 \\
    --min_lr {min_lr} \\
    --eval_interval 40 \\
    --max_length {max_length} \\
    --num_workers 8 \\
    --dataset_num_proc 8 \\
    --no_save_optim true \\
    --no_save_rng true \\
    --sequence_parallel true \\
    --attention_backend flash \\
    --optimizer_cpu_offload true \\
    --use_precision_aware_optimizer true \\
    --merge_lora false \\
    --use_hf 1
"""
        
        return FrameworkOutput(
            runtime=Runtime(
                start_commands=["chmod +x run.sh && ./run.sh"],
                environment_variables=get_environment_variables(sft_config),
                cache_config=CacheConfig(enabled=True),
                checkpointing_config=CheckpointingConfig(enabled=True),
            ),
            train_script=run_script,  # This will be written to train.py, we need to write to run.sh
            base_image="baseten/megatron:py3.11.11-cuda12.8.1-torch2.8.0-fa2.8.1-megatron0.14.1-msswift3.10.3",
        )
