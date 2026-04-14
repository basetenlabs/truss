#!/bin/bash
set -eux

# --- Install dependencies ---
source "$(dirname "${BASH_SOURCE[0]}")/init.sh"

# --- Configuration ---
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
DATASET="${DATASET:-AI-MO/NuminaMath-TIR#1000}"

export HF_HOME="${BT_RW_CACHE_DIR}/huggingface"
mkdir -p "${HF_HOME}"

SHARED="${BT_RW_CACHE_DIR}/${BT_TRAINING_JOB_NAME}"
mkdir -p "${SHARED}"

ROLLOUT_NODE=$((BT_GROUP_SIZE - 1))
ROLLOUT_PORT=8000
MY_IP=$(hostname -I | awk '{print $1}')

# Register this node's IP
echo "$MY_IP" > "${SHARED}/node_${BT_NODE_RANK}.ip"

# --- Node roles ---
# Last node = rollout server, all others = training

if [ "$BT_NODE_RANK" -eq "$ROLLOUT_NODE" ]; then
    # ==================== ROLLOUT NODE ====================
    echo "=== Node ${BT_NODE_RANK}: Rollout Server ==="

    # Download model to shared cache
    python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"
    touch "${SHARED}/.model_ready"
    echo "$MY_IP" > "${SHARED}/rollout_host.ip"

    # Start vLLM rollout server
    swift rollout \
        --model "${MODEL_ID}" \
        --use_hf true \
        --vllm_tensor_parallel_size 2 \
        --vllm_gpu_memory_utilization 0.9 \
        --port ${ROLLOUT_PORT} &
    ROLLOUT_PID=$!

    # Wait for server health
    waited=0
    until curl -s "http://127.0.0.1:${ROLLOUT_PORT}/health" >/dev/null 2>&1; do
        sleep 5; waited=$((waited + 5))
        kill -0 $ROLLOUT_PID 2>/dev/null || { echo "Rollout died"; exit 1; }
        [ $waited -lt 600 ] || { echo "Rollout timeout"; exit 1; }
    done
    touch "${SHARED}/.rollout_ready"
    echo "Rollout server ready"

    # Wait for training to finish
    while [ ! -f "${SHARED}/.training_complete" ]; do sleep 30; done
    kill $ROLLOUT_PID 2>/dev/null || true

else
    # ==================== TRAINING NODE ====================
    echo "=== Node ${BT_NODE_RANK}: Training ==="

    # Wait for rollout
    while [ ! -f "${SHARED}/.model_ready" ]; do sleep 5; done
    while [ ! -f "${SHARED}/.rollout_ready" ]; do sleep 5; done
    sleep 5  # grace period

    ROLLOUT_HOST=$(cat "${SHARED}/rollout_host.ip")
    MASTER_ADDR=$(cat "${SHARED}/node_0.ip")
    TRAINING_NNODES=$((BT_GROUP_SIZE - 1))
    DP=$((BT_NUM_GPUS * TRAINING_NNODES))

    echo "Training: NNODES=${TRAINING_NNODES} RANK=${BT_NODE_RANK} MASTER=${MASTER_ADDR} ROLLOUT=${ROLLOUT_HOST}:${ROLLOUT_PORT}"

    PYTORCH_ALLOC_CONF=expandable_segments:True \
    NNODES=$TRAINING_NNODES \
    NODE_RANK=$BT_NODE_RANK \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=29500 \
    NPROC_PER_NODE=$BT_NUM_GPUS \
    megatron rlhf \
        --rlhf_type grpo \
        --model "${MODEL_ID}" \
        --use_hf true \
        --dataset "${DATASET}" \
        --reward_funcs accuracy \
        --use_vllm true \
        --vllm_mode server \
        --vllm_server_host "${ROLLOUT_HOST}" \
        --vllm_server_port ${ROLLOUT_PORT} \
        --tensor_model_parallel_size 1 \
        --pipeline_model_parallel_size 1 \
        --context_parallel_size 1 \
        --micro_batch_size 1 \
        --global_batch_size $((DP * 2)) \
        --steps_per_generation 4 \
        --num_generations 8 \
        --max_length 4096 \
        --max_completion_length 2048 \
        --temperature 0.6 \
        --tuner_type full \
        --lr 1e-6 \
        --bf16 true \
        --beta 0.001 \
        --loss_type grpo \
        --epsilon 0.2 \
        --finetune \
        --max_epochs 1 \
        --log_interval 1 \
        --log_completions true \
        --save "${SHARED}/checkpoints" \
        --save_interval 100 \
        --report_to wandb

    [ "$BT_NODE_RANK" -eq 0 ] && touch "${SHARED}/.training_complete"
fi

# Sync checkpoints
rsync -ah "${SHARED}/" "${BT_CHECKPOINT_DIR}/" 2>/dev/null || true
echo "Node ${BT_NODE_RANK} done"
