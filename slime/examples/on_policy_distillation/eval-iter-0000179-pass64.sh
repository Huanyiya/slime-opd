#!/bin/bash

# Evaluate checkpoint iter_0000179 on AIME24, AIME25, and AMC23.
# Each prompt is sampled 64 times. Results are written to JSON.
#
# Usage:
#   bash examples/on_policy_distillation/eval-iter-0000179-pass64.sh

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -euo pipefail
set -x

REPO_ROOT="/mnt/cpfs/users/zhy/opd/slime-OPD"
SLIME_DIR="${REPO_ROOT}/slime"
SLIME_PYTHON="${REPO_ROOT}/.venv/bin/python"
MEGATRON_PATH="${REPO_ROOT}/Megatron-LM"
HF_CHECKPOINT="/mnt/cpfs/weights/Qwen3.5-4B"
CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints/08_01_22_35_student_Qwen3.5-4B_teacher_Qwen3.5-9B"
CHECKPOINT_STEP=179
RESULT_PATH="${REPO_ROOT}/eval_iter_0000179_pass64.json"

if [ ! -f "${CHECKPOINT_ROOT}/iter_0000179/.metadata" ]; then
    echo "Missing torch_dist checkpoint: ${CHECKPOINT_ROOT}/iter_0000179" >&2
    exit 1
fi

LOCAL_NO_PROXY="127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"
export RAY_DISABLE_DASHBOARD_GPU_METRICS=1
export PYTHONUNBUFFERED=1
export OPD_EVAL_RESULT_PATH="${RESULT_PATH}"
export FLASHINFER_WORKSPACE_BASE="/tmp/slime_eval_flashinfer"

cd "${SLIME_DIR}"
source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

# This script owns the local eight-GPU Ray cluster used for evaluation.
"${SLIME_PYTHON}" -m ray.scripts.scripts stop --force || true

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
"${SLIME_PYTHON}" -m ray.scripts.scripts start \
    --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 8 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

RAY_DASHBOARD_URL="http://127.0.0.1:8265"
RAY_DASHBOARD_READY=0
for _ in $(seq 1 60); do
    if curl -sf "${RAY_DASHBOARD_URL}/api/version" >/dev/null; then
        RAY_DASHBOARD_READY=1
        break
    fi
    sleep 1
done
if [ "${RAY_DASHBOARD_READY}" -ne 1 ]; then
    echo "Ray Dashboard failed to start at ${RAY_DASHBOARD_URL}" >&2
    exit 1
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${no_proxy}\",
    \"FLASHINFER_WORKSPACE_BASE\": \"${FLASHINFER_WORKSPACE_BASE}\",
    \"OPD_EVAL_RESULT_PATH\": \"${RESULT_PATH}\"
  }
}"

"${SLIME_PYTHON}" -m ray.scripts.scripts job submit \
    --address="${RAY_DASHBOARD_URL}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${SLIME_PYTHON}" train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --rollout-num-gpus 6 \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --load "${CHECKPOINT_ROOT}" \
    --ckpt-step "${CHECKPOINT_STEP}" \
    --no-load-optim \
    --no-load-rng \
    --num-rollout 0 \
    --prompt-data /mnt/cpfs/users/zhy/opd/OPD/datasets/dapo-math-17k-processed.parquet \
    --input-key prompt \
    --apply-chat-template \
    --apply-chat-template-kwargs '{"enable_thinking": false}' \
    --rollout-batch-size 16 \
    --n-samples-per-prompt 8 \
    --rollout-max-response-len 8192 \
    --global-batch-size 64 \
    --advantage-estimator grpo \
    --optimizer adam \
    --lr 1e-6 \
    --tensor-model-parallel-size 1 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 8192 \
    --log-probs-chunk-size 2048 \
    --eval-interval 1 \
    --eval-prompt-data \
        AIME24 /mnt/cpfs/users/zhy/opd/OPD/datasets/test_data/AIME24/test.parquet \
        AIME25 /mnt/cpfs/users/zhy/opd/OPD/datasets/test_data/AIME25/test.parquet \
        AMC23 /mnt/cpfs/users/zhy/opd/OPD/datasets/test_data/AMC23/test.parquet \
    --eval-input-key prompt \
    --eval-label-key reward_model \
    --eval-custom-rm-path slime.rollout.on_policy_distillation.math_eval_reward_func \
    --custom-eval-rollout-log-function-path slime.rollout.on_policy_distillation.log_eval_metrics_64 \
    --n-samples-per-eval-prompt 64 \
    --eval-temperature 1 \
    --eval-top-p 0.95 \
    --eval-max-response-len 8192 \
    --custom-generate-function-path slime.rollout.on_policy_distillation.boxed_prompt_generate_func \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static 0.6 \
    --sglang-server-concurrency 64 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash

echo "Evaluation results: ${RESULT_PATH}"
