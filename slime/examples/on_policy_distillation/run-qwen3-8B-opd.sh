#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd.sh

if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -ex

# Ensure Ray, SGLang, and the submitted training job do not inherit HTTP(S) proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# torch_memory_saver backs colocated SGLang offload/onload and is incompatible
# with PyTorch's expandable-segments allocator.
unset PYTORCH_CUDA_ALLOC_CONF

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LOCAL_NO_PROXY="127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"
# HGGC does not implement the NVIDIA per-process utilization API used by Ray's
# dashboard agent. This disables dashboard-only GPU telemetry; Ray GPU resource
# scheduling and the Slime/SGLang workloads are unaffected.
export RAY_DISABLE_DASHBOARD_GPU_METRICS=1
# Keep launcher/status output quiet and preserve one copy of every worker's
# fatal/crash diagnostics during the next run.
export NUMEXPR_MAX_THREADS=64

STUDENT_MODEL_PATH="/mnt/cpfs/weights/Qwen3.5-4B"
TEACHER_MODEL_PATH="/mnt/cpfs/weights/Qwen3.5-9B"
SLIME_PYTHON="/mnt/cpfs/users/zhy/opd/slime-OPD/.venv/bin/python"
MEGATRON_PATH="/mnt/cpfs/users/zhy/opd/slime-OPD/Megatron-LM"
STUDENT_MODEL_NAME="$(basename "${STUDENT_MODEL_PATH}")"
TEACHER_MODEL_NAME="$(basename "${TEACHER_MODEL_PATH}")"
OPD_TOP_K=16
SEED=42

CHECKPOINT_ROOT="/mnt/cpfs/users/zhy/opd/slime-OPD/checkpoints"
RUN_TIMESTAMP=$(TZ=Asia/Shanghai date '+%m_%d_%H_%M')
RUN_NAME="${RUN_TIMESTAMP}_student_${STUDENT_MODEL_NAME}_teacher_${TEACHER_MODEL_NAME}"
RUN_CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
RUN_LOG_DIR="${REPO_ROOT}/run_logs/${RUN_NAME}"
# Compiler and Ray runtime files are transient. Eight-rank compilation can
# create tens of GiB, so keep them in the large local RAM filesystem rather
# than in the 30 GiB /tmp filesystem or a concurrent network filesystem.
SCRATCH_ROOT="/dev/shm/slime-opd-${RUN_TIMESTAMP}-$$"
# Keep compiler artifacts across runs on the node.  A per-run cache forces
# Triton/Inductor to compile the same kernels again after every restart; /dev/shm
# avoids the container-overlay stale-handle issue seen with /tmp.
COMPILE_CACHE_DIR="/dev/shm/slime-opd-persistent-cache"
# Ray appends a long session name and dashboard socket name. Keep its root
# deliberately short so AF_UNIX socket paths stay below Linux's 107-byte cap.
RAY_TMP_DIR="/dev/shm/ray-$$"
export TMPDIR="${SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export TRITON_CACHE_DIR="${COMPILE_CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${COMPILE_CACHE_DIR}/torchinductor"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${RAY_TMP_DIR}"
# Avoid the colocated CUDA-IPC weight path: concurrent CUDA tensor export from
# eight Megatron ranks crashes the HGGC driver on this PPU.  Do not put the HF
# shards in /tmp either: the local filesystem is only about 30 GiB and Ray uses
# it at the same time, which can fill it halfway through the 16-shard export.
WEIGHT_SYNC_ROOT="${CHECKPOINT_ROOT}/.weight-sync"
WEIGHT_SYNC_DIR="${WEIGHT_SYNC_ROOT}/${RUN_NAME}-$$"
DIAGNOSTICS_DIR="${RUN_LOG_DIR}/diagnostics"
JOB_CONSOLE_LOG="${RUN_LOG_DIR}/job-console.log"
RAY_STARTED=0

mkdir -p "${CHECKPOINT_ROOT}" "${RUN_LOG_DIR}" "${WEIGHT_SYNC_DIR}"
if [ -e "${RUN_CHECKPOINT_DIR}" ]; then
    echo "Checkpoint directory already exists: ${RUN_CHECKPOINT_DIR}" >&2
    exit 1
fi
mkdir "${RUN_CHECKPOINT_DIR}"
echo "Student checkpoints will be saved to: ${RUN_CHECKPOINT_DIR}"
echo "Console and crash logs will be saved to: ${RUN_LOG_DIR}"
cp "${BASH_SOURCE[0]}" "${RUN_LOG_DIR}/launch-script.sh"
printf '%s\n' "${RAY_TMP_DIR}" > "${RUN_LOG_DIR}/ray-temp-dir.txt"
printf '%s\n' "${WEIGHT_SYNC_DIR}" > "${RUN_LOG_DIR}/weight-sync-dir.txt"
printf '%s\n' "${SCRATCH_ROOT}" > "${RUN_LOG_DIR}/scratch-dir.txt"

# Save the Ray logs before cleanup.  A raylet that is killed externally often
# leaves no Python traceback, so these logs are the primary failure evidence.
save_ray_diagnostics() {
   mkdir -p "${DIAGNOSTICS_DIR}"
   if [ -d "${RAY_TMP_DIR}" ]; then
      cp -a "${RAY_TMP_DIR}" "${DIAGNOSTICS_DIR}/ray" 2>/dev/null || true
   fi
   dmesg -T 2>/dev/null | tail -n 500 > "${DIAGNOSTICS_DIR}/dmesg-tail.log" || true
   ps -eo pid,ppid,lstart,stat,pcpu,pmem,args > "${DIAGNOSTICS_DIR}/processes.log" 2>&1 || true
   nvidia-smi > "${DIAGNOSTICS_DIR}/nvidia-smi.log" 2>&1 || true
   echo "Saved run diagnostics to: ${DIAGNOSTICS_DIR}"
}

# Only terminate processes that explicitly belong to this run's unique Ray
# temp directory.  Do not use global pkill: it can kill a different run's
# raylet/worker and make that run look like a random Ray failure.
stop_this_ray_cluster() {
   [ "${RAY_STARTED}" -eq 1 ] || return 0
   local ray_pids
   ray_pids=$(ps -eo pid=,args= | awk -v marker="${RAY_TMP_DIR}" 'index($0, marker) {print $1}')
   if [ -n "${ray_pids}" ]; then
      echo "Stopping Ray processes for this run: ${ray_pids}"
      kill -TERM ${ray_pids} 2>/dev/null || true
      sleep 3
      ray_pids=$(ps -eo pid=,args= | awk -v marker="${RAY_TMP_DIR}" 'index($0, marker) {print $1}')
      [ -z "${ray_pids}" ] || kill -KILL ${ray_pids} 2>/dev/null || true
   fi
}

cleanup() {
   local exit_code=$?
   trap - EXIT
   # The terminal stream is always persisted in RUN_LOG_DIR. Copy the larger
   # Ray worker logs only after a failed run, when they are needed for debug.
   if [ "${exit_code}" -ne 0 ]; then
      save_ray_diagnostics
   fi
   stop_this_ray_cluster
   # Remove only this run's partial/temporary HF export.  Successful updates
   # normally remove their version subdirectories themselves; this also covers
   # an interrupted export without touching checkpoints from any other run.
   if [[ "${WEIGHT_SYNC_DIR}" == "${WEIGHT_SYNC_ROOT}/"* ]] && [ -d "${WEIGHT_SYNC_DIR}" ]; then
      rm -rf -- "${WEIGHT_SYNC_DIR}"
   fi
# Per-run Ray runtime files are never retained.  COMPILE_CACHE_DIR is
# intentionally excluded: it is shared across runs to reuse compiled kernels.
   if [[ "${SCRATCH_ROOT}" == /dev/shm/slime-opd-* ]] && [ -d "${SCRATCH_ROOT}" ]; then
      rm -rf -- "${SCRATCH_ROOT}"
   fi
   if [[ "${RAY_TMP_DIR}" == /dev/shm/ray-* ]] && [ -d "${RAY_TMP_DIR}" ]; then
      rm -rf -- "${RAY_TMP_DIR}"
   fi
   exit "${exit_code}"
}
trap cleanup EXIT

# CHECKPOINT_ROOT="/mnt/cpfs/users/zhy/opd/slime-OPD/checkpoints"
# RUN_CHECKPOINT_DIR="${CHECKPOINT_ROOT}/08_04_15_24_student_Qwen3.5-4B_teacher_Qwen3.5-9B"

# if [ ! -f "${RUN_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
#     echo "Missing latest_checkpointed_iteration.txt in ${RUN_CHECKPOINT_DIR}" >&2
#     exit 1
# fi

# echo "Resume training from: ${RUN_CHECKPOINT_DIR}"

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/mnt/cpfs/users/zhy/opd/slime-OPD/slime/scripts/models/qwen3.5-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_MODEL_PATH}"
   --ref-load "${CHECKPOINT_ROOT}/${STUDENT_MODEL_NAME}_torch_dist"
   --load "${RUN_CHECKPOINT_DIR}"
   --save "${RUN_CHECKPOINT_DIR}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/cpfs/users/zhy/opd/OPD/datasets/dapo-math-17k-processed.parquet
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --rollout-seed "${SEED}"
   --num-rollout 300
   --rollout-batch-size 64
   --n-samples-per-prompt 4
   --rollout-max-response-len 16384
   --rollout-temperature 1
   --apply-chat-template-kwargs '{"enable_thinking": false}'
   --global-batch-size 256
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --sequential-opd-teacher-model-path "${TEACHER_MODEL_PATH}"
   --sequential-opd-teacher-num-gpus-per-engine 1
   --sequential-opd-teacher-mem-fraction-static 0.7
   --sequential-opd-teacher-chunked-prefill-size 6144
)

EVAL_ARGS=(
   --eval-interval 5000
   --skip-eval-before-train
   --eval-prompt-data \
      AIME24 /mnt/cpfs/users/zhy/opd/OPD/datasets/test_data/AIME24/test.parquet \
      AIME25 /mnt/cpfs/users/zhy/opd/OPD/datasets/test_data/AIME25/test.parquet \
      AMC23 /mnt/cpfs/users/zhy/opd/OPD/datasets/test_data/AMC23/test.parquet
   --eval-input-key prompt
   --eval-label-key reward_model
   --eval-custom-rm-path slime.rollout.on_policy_distillation.math_eval_reward_func
   --custom-eval-rollout-log-function-path slime.rollout.on_policy_distillation.log_eval_metrics
   --n-samples-per-eval-prompt 16
   --log-passrate
   --eval-temperature 0.7
   --eval-top-p 0.8
   --eval-top-k 20
   --eval-min-p 0.0
   --eval-presence-penalty 1.5
   --eval-repetition-penalty 1.0
   --eval-max-response-len 16384
)

PERF_ARGS=(
   --log-opd-phase-times-only

   --update-weight-mode full
   --update-weight-transport disk
   --update-weight-disk-dir "${WEIGHT_SYNC_DIR}"

   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 12288
   --log-probs-chunk-size 4096
)
#   --opd-k1-diff-clip 10.0
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-loss-type topk
   --opd-top-k "${OPD_TOP_K}"
   --opd-kl-coef 1.0
   --opd-kl-loss-type k1
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
# OPTIMIZER_ARGS=(
#    --optimizer adam

#    --lr 1e-6
#    --lr-decay-style cosine
#    --min-lr 1e-7
#    --lr-warmup-fraction 0.05
#    --lr-decay-iters 150

#    --weight-decay 0.1
#    --adam-beta1 0.9
#    --adam-beta2 0.98
# )
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-opd
   --wandb-group qwen3.5-4B-opd-qwen3.5-9B-topk-bsz64-clip10-renormalization
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 64
   --sglang-num-continuous-decode-steps 8
   # The router starts before the 8 SGLang workers register. power_of_two
   # rejects an initially empty worker list; round_robin starts empty and then
   # distributes rollout requests evenly after registration.
   --router-policy round_robin
)


MISC_ARGS=(
   --seed "${SEED}"
   --custom-generate-function-path slime.rollout.on_policy_distillation.boxed_prompt_generate_func
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)




# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
if curl -sf --connect-timeout 2 --max-time 5 "http://127.0.0.1:8265/api/version" >/dev/null; then
   echo "A Ray dashboard is already listening on port 8265; refusing to stop it because it may belong to another run." >&2
   exit 1
fi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
"${SLIME_PYTHON}" -m ray.scripts.scripts start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir "${RAY_TMP_DIR}"
RAY_STARTED=1

RAY_DASHBOARD_URL="http://127.0.0.1:8265"
RAY_DASHBOARD_READY=0
for _ in $(seq 1 60); do
   if curl -sf --connect-timeout 2 --max-time 5 "${RAY_DASHBOARD_URL}/api/version" >/dev/null; then
      RAY_DASHBOARD_READY=1
      break
   fi
   sleep 1
done
if [ "${RAY_DASHBOARD_READY}" -ne 1 ]; then
   echo "Ray Dashboard failed to start at ${RAY_DASHBOARD_URL}" >&2
   tail -n 100 "${RAY_TMP_DIR}"/session_latest/logs/dashboard.log 2>/dev/null || true
   tail -n 100 "${RAY_TMP_DIR}"/session_latest/logs/dashboard.err 2>/dev/null || true
   exit 1
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"TMPDIR\": \"${TMPDIR}\",
    \"TMP\": \"${TMP}\",
    \"TEMP\": \"${TEMP}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"RELAX_OPD_TOKEN_IDS_LOGPROB_K\": \"${OPD_TOP_K}\",
    \"PYTHONFAULTHANDLER\": \"1\",
    \"TORCH_SHOW_CPP_STACKTRACES\": \"1\",
    \"TORCH_DISABLE_ADDR2LINE\": \"1\",
    \"NCCL_DEBUG\": \"WARN\",
    \"SLIME_RELOAD_PROCESS_GROUPS\": \"0\",
    \"RAY_DEDUP_LOGS\": \"1\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${no_proxy}\"
  }
}"

JOB_SUBMISSION_ID="slime-opd-${RUN_TIMESTAMP//_/-}-$$"
"${SLIME_PYTHON}" -m ray.scripts.scripts job submit --address="${RAY_DASHBOARD_URL}" \
   --submission-id="${JOB_SUBMISSION_ID}" \
   --no-wait \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${SLIME_PYTHON}" train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${RM_ARGS[@]}"

# Stream the complete Ray job output to both the terminal and a persistent
# repository-local log. If the websocket drops while the job is still alive,
# query the authoritative status and reconnect instead of cleaning up Ray.
while true; do
   echo "Following Ray job logs for ${JOB_SUBMISSION_ID}; also writing ${JOB_CONSOLE_LOG}"
   set +e
   "${SLIME_PYTHON}" -m ray.scripts.scripts job logs \
      --address="${RAY_DASHBOARD_URL}" --follow "${JOB_SUBMISSION_ID}" 2>&1 | tee -a "${JOB_CONSOLE_LOG}"
   JOB_LOG_EXIT_CODE=${PIPESTATUS[0]}
   set -e

   if ! JOB_STATUS=$("${SLIME_PYTHON}" -m ray.scripts.scripts job status \
      --address="${RAY_DASHBOARD_URL}" "${JOB_SUBMISSION_ID}" 2>&1); then
      echo "Ray log stream exited with code ${JOB_LOG_EXIT_CODE}; status query failed, retrying in 5 seconds:" >&2
      echo "${JOB_STATUS}" >&2
      sleep 5
      continue
   fi
   printf '%s\n' "${JOB_STATUS}" > "${RUN_LOG_DIR}/final-job-status.txt"
   echo "Ray job ${JOB_SUBMISSION_ID} status: ${JOB_STATUS}"
   JOB_STATE=$(printf '%s\n' "${JOB_STATUS}" | tr '[:lower:]' '[:upper:]')
   case "${JOB_STATE}" in
      *SUCCEEDED*)
         break
         ;;
      *FAILED*|*STOPPED*)
         echo "Ray job ${JOB_SUBMISSION_ID} ended unsuccessfully." >&2
         exit 1
         ;;
      *PENDING*|*RUNNING*)
         echo "Ray log stream disconnected while the job is still active; reconnecting in 5 seconds." >&2
         sleep 5
         ;;
      *)
         echo "Unrecognized Ray job status; reconnecting logs in 5 seconds." >&2
         sleep 5
         ;;
   esac
done
