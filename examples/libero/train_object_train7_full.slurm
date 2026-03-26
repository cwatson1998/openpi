#!/bin/bash
#SBATCH --job-name=openpi_object_train7_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

set -euo pipefail

# If your cluster requires an account or partition, pass them at submission time, for example:
# sbatch --partition=gpu --account=YOUR_ACCOUNT examples/libero/train_object_train7_full.slurm

cd "${SLURM_SUBMIT_DIR:-/home/christopher/Documents/openpi-finetune/openpi}"

export USE_TF=0
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export TOKENIZERS_PARALLELISM=false

CONFIG_NAME="${CONFIG_NAME:-pi0_fast_libero_object_train7}"
EXP_NAME="${EXP_NAME:-object_train7_full_slurm}"
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

mkdir -p "$(dirname "${CHECKPOINT_DIR}")"

echo "[$(timestamp)] Host: $(hostname)"
echo "[$(timestamp)] Starting full fine-tune"
echo "[$(timestamp)] Config: ${CONFIG_NAME}"
echo "[$(timestamp)] Experiment: ${EXP_NAME}"
echo "[$(timestamp)] Checkpoints: ${CHECKPOINT_DIR}"
echo "[$(timestamp)] Resume mode enabled"

CMD=(
  uv run scripts/train.py "${CONFIG_NAME}"
  --exp-name="${EXP_NAME}"
  --resume
  --save-interval="${SAVE_INTERVAL}"
  --keep-period="${KEEP_PERIOD}"
  --num-workers="${NUM_WORKERS}"
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
