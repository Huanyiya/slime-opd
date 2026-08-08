#!/bin/bash

set -u
set -o pipefail

# Do not inherit HTTP(S) proxy settings into Ray, SGLang, or the training job.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# =========================
# Configuration
# =========================

WORK_DIR="/mnt/cpfs/users/zhy/opd/slime-OPD"

# 修改为你实际的训练脚本路径
TRAIN_SCRIPT="${WORK_DIR}/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh"

VENV_ACTIVATE="${WORK_DIR}/.venv/bin/activate"

LOG_DIR="${WORK_DIR}/wait_train_logs"

# 每隔多少秒检查一次
CHECK_INTERVAL=60

# 连续多少次检测为空闲后才启动
REQUIRED_IDLE_CHECKS=2

GPU_IDS=(0 1 2 3 4 5 6 7)

# 防止重复启动多个 watcher
LOCK_FILE="/tmp/zhy_slime_opd_gpu_wait.lock"

mkdir -p "${LOG_DIR}"

# =========================
# Prevent duplicate watcher
# =========================

exec 9>"${LOCK_FILE}"

if ! flock -n 9; then
    echo "Another GPU waiting script is already running."
    echo "Lock file: ${LOCK_FILE}"
    exit 1
fi

# =========================
# Check files
# =========================

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Training script does not exist:"
    echo "${TRAIN_SCRIPT}"
    exit 1
fi

if [ ! -f "${VENV_ACTIVATE}" ]; then
    echo "Virtual environment does not exist:"
    echo "${VENV_ACTIVATE}"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found."
    exit 1
fi

# =========================
# GPU check function
# =========================

all_gpus_idle() {
    local gpu_id
    local gpu_processes

    for gpu_id in "${GPU_IDS[@]}"; do
        # 确认 GPU 存在
        if ! nvidia-smi \
            -i "${gpu_id}" \
            --query-gpu=index \
            --format=csv,noheader,nounits \
            >/dev/null 2>&1; then

            echo "GPU ${gpu_id} does not exist or nvidia-smi failed."
            return 1
        fi

        # 查询这张 GPU 上的 CUDA compute process
        gpu_processes=$(
            nvidia-smi \
                -i "${gpu_id}" \
                --query-compute-apps=pid \
                --format=csv,noheader,nounits \
                2>/dev/null |
            sed '/^[[:space:]]*$/d'
        )

        if [ -n "${gpu_processes}" ]; then
            return 1
        fi
    done

    return 0
}

# =========================
# Wait for idle GPUs
# =========================

echo "=================================================="
echo "Waiting for GPU 0-7 to become completely idle"
echo "Check interval: ${CHECK_INTERVAL} seconds"
echo "Required consecutive idle checks: ${REQUIRED_IDLE_CHECKS}"
echo "Training script: ${TRAIN_SCRIPT}"
echo "=================================================="

idle_count=0

while true; do
    current_time=$(date '+%Y-%m-%d %H:%M:%S')

    if all_gpus_idle; then
        idle_count=$((idle_count + 1))

        echo "[${current_time}] GPU 0-7 idle: ${idle_count}/${REQUIRED_IDLE_CHECKS}"

        if [ "${idle_count}" -ge "${REQUIRED_IDLE_CHECKS}" ]; then
            echo "[${current_time}] All GPUs are idle. Starting training."
            break
        fi
    else
        if [ "${idle_count}" -gt 0 ]; then
            echo "[${current_time}] GPU became busy again. Reset idle counter."
        else
            echo "[${current_time}] Some GPUs are still busy."
        fi

        idle_count=0

        # 打印当前 GPU 进程，方便观察
        nvidia-smi \
            --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null || true
    fi

    sleep "${CHECK_INTERVAL}"
done

# =========================
# Start training
# =========================

source "${VENV_ACTIVATE}"
cd "${WORK_DIR}" || exit 1

START_TIME=$(date '+%Y-%m-%d_%H-%M-%S')
TRAIN_LOG="${LOG_DIR}/opd_train_${START_TIME}.log"

echo "Virtual environment: ${VIRTUAL_ENV:-unknown}"
echo "Training log: ${TRAIN_LOG}"
echo "Training started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

# 输出既显示在当前终端，也保存到日志
bash "${TRAIN_SCRIPT}" 2>&1 | tee -a "${TRAIN_LOG}"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo "=================================================="
echo "Training finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Training exit code: ${TRAIN_EXIT_CODE}"
echo "Training log: ${TRAIN_LOG}"

exit "${TRAIN_EXIT_CODE}"
