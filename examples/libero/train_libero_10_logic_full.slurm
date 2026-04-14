#!/bin/bash
#SBATCH --job-name=openpi_libero10_logic_full
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=dj-l40-0.grasp.maas,dj-a40-1.grasp.maas
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/chriswatson/openpi-finetune/openpi}"

export USE_TF=0
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

CONFIG_NAME="${CONFIG_NAME:-pi0_fast_libero_10_logic}"
EXP_NAME="${EXP_NAME:-libero10_logic_full}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
KEEP_PERIOD="${KEEP_PERIOD:-1000}"

CHECKPOINT_DIR="checkpoints/${CONFIG_NAME}/${EXP_NAME}"
TRAIN_PID=""
REQUEUED_ALREADY=0

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

requeue_job() {
  if [[ -n "${SLURM_JOB_ID:-}" && "${REQUEUED_ALREADY}" -eq 0 ]]; then
    REQUEUED_ALREADY=1
    echo "[$(timestamp)] Requeuing job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
  fi
}

handle_signal() {
  local signal_name="$1"
  echo "[$(timestamp)] Caught ${signal_name}. Requesting requeue and stopping training."
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" || true
  fi
  requeue_job
  exit 0
}

trap 'handle_signal USR1' USR1
trap 'handle_signal TERM' TERM

mkdir -p "$(dirname "${CHECKPOINT_DIR}")" logs/slurm

echo "[$(timestamp)] Host: $(hostname)"
echo "[$(timestamp)] Starting full fine-tune"
echo "[$(timestamp)] Config: ${CONFIG_NAME}"
echo "[$(timestamp)] Experiment: ${EXP_NAME}"
echo "[$(timestamp)] Checkpoints: ${CHECKPOINT_DIR}"
echo "[$(timestamp)] FSDP devices: ${FSDP_DEVICES}"
echo "[$(timestamp)] Resume mode enabled"

CMD=(
  uv run --frozen --no-dev scripts/train.py "${CONFIG_NAME}"
  --exp-name="${EXP_NAME}"
  --fsdp-devices="${FSDP_DEVICES}"
  --resume
  --save-interval="${SAVE_INTERVAL}"
  --keep-period="${KEEP_PERIOD}"
  --num-workers="${NUM_WORKERS}"
  --no-wandb-enabled
)

echo "[$(timestamp)] Command: ${CMD[*]}"
"${CMD[@]}" &
TRAIN_PID=$!
set +e
wait "${TRAIN_PID}"
EXIT_CODE=$?
set -e

if [[ "${EXIT_CODE}" -eq 0 ]]; then
  echo "[$(timestamp)] Training finished successfully."
  exit 0
fi

if [[ "${EXIT_CODE}" -eq 143 || "${EXIT_CODE}" -eq 130 ]]; then
  echo "[$(timestamp)] Training exited due to a termination signal."
  requeue_job
  exit 0
fi

echo "[$(timestamp)] Training failed with exit code ${EXIT_CODE}."
exit "${EXIT_CODE}"
