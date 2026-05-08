# !/bin/bash

# Qwen3.5-122B-A10B VL RL training on geo3k dataset
# Colocate 模式：8 节点 64 GPU，训练和推理共用

export https_proxy=http://10.3.4.34:3128
export http_proxy=http://10.3.4.34:3128
export SGLANG_NUMA_BIND_V2=0
export SGLANG_DISABLE_CUDNN_CHECK=1
export NCCL_DEBUG=WARN

# Configuration
TRAIN_BACKEND="megatron"
MODEL_NAME="Qwen3.5-122B-A10B"
DATASET_NAME=${SLIME_SCRIPT_DATASET_NAME:-"chenhegu/geo3k_imgurl"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
  USE_EXTERNAL_RAY=0
else
  USE_EXTERNAL_RAY=1
fi

# Cleanup
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
  ray stop --force
  pkill -9 ray
fi
pkill -9 slime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
  pkill -9 ray
fi
pkill -9 slime
pkill -9 redis

set -ex

export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Download dataset
DATASET_BASE_DIR="/mnt/tidal-alsh01/dataset/redone/mmx/slime-zr/datasets"
mkdir -p "${DATASET_BASE_DIR}"
if [ ! -d "${DATASET_BASE_DIR}/${DATASET_LOCAL_NAME}" ]; then
  echo "Downloading dataset to NAS: ${DATASET_BASE_DIR}/${DATASET_LOCAL_NAME}"
  hf download --repo-type dataset ${DATASET_NAME} --local-dir ${DATASET_BASE_DIR}/${DATASET_LOCAL_NAME}
fi

# Common args
CKPT_ARGS=(
  --hf-checkpoint /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/${MODEL_NAME}
  --load /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/${MODEL_NAME}
  --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
  --prompt-data ${DATASET_BASE_DIR}/${DATASET_LOCAL_NAME}/train.parquet
  --input-key problem
  --label-key answer
  --apply-chat-template
  --rollout-shuffle
  --rm-type deepscaler
  --num-rollout 3000
  --rollout-batch-size 32
  --n-samples-per-prompt 4
  --rollout-max-response-len 4096
  --rollout-temperature 0.8
  --global-batch-size 128
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'

EVAL_ARGS=(
  --eval-interval 20
  --eval-prompt-data ${DATASET_LOCAL_NAME} ${DATASET_BASE_DIR}/${DATASET_LOCAL_NAME}/test.parquet
  --n-samples-per-eval-prompt 1
  --eval-max-response-len 4096
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 8
  --sglang-mem-fraction-static 0.40
  --sglang-ep-size 8
  --disable-cuda-graph
  --sglang-max-running-requests 256
)

# Wandb args
if [ -n "$WANDB_API_KEY" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project slime-geo3k-vlm
    --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
    --wandb-key ${WANDB_API_KEY}
    --disable-wandb-random-suffix
  )
else
  WANDB_ARGS=()
fi

MISC_ARGS=(
  --colocate
  # --train-memory-margin-bytes 1073741824
)

BACKEND_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 8
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --qkv-format bshd
  --micro-batch-size 1
)

SLIME_DIR="/mnt/tidal-alsh01/dataset/redone/mmx/slime-zr/slime"

source "${SLIME_DIR}/scripts/models/qwen3.5-122B-A10B.sh"

if [ "$USE_EXTERNAL_RAY" = "0" ]; then
  if [ -z "${MASTER_ADDR}" ]; then
    echo "MASTER_ADDR is not set (should be injected by K8s scheduler)"
    exit 1
  fi
  export no_proxy="127.0.0.1,${MASTER_ADDR}"

  if [ "${RANK:-0}" = "0" ]; then
    echo "Starting Ray HEAD on ${HOST_IP}"
    ray start --head --node-ip-address ${HOST_IP} --num-gpus ${NUM_GPUS} \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
  else
    echo "Starting Ray WORKER on ${HOST_IP}, connecting to ${MASTER_ADDR}:6379"
    ray start --address=${MASTER_ADDR}:6379 --num-gpus ${NUM_GPUS} \
      --node-ip-address ${HOST_IP} --disable-usage-stats
  fi
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

# 只在主节点提交 job
if [ "${RANK:-0}" != "0" ]; then
  echo "Worker node (RANK=${RANK}): Ray started, waiting for job from master..."
  wait
  exit 0
fi

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 8 \
  --actor-num-gpus-per-node ${NUM_GPUS} \
  --multimodal-keys "${MULTIMODAL_KEYS}" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${BACKEND_ARGS[@]} \
  ${MISC_ARGS[@]} 2>&1 | tee "training_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
