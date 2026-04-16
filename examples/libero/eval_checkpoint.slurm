#!/bin/bash
#SBATCH --job-name=openpi_libero_eval
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dj-a40-0.grasp.maas,dj-a40-1.grasp.maas,dj-l40-0.grasp.maas
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/chriswatson/openpi-finetune/openpi}"
cd "$ROOT_DIR"

POLICY_CONFIG="${POLICY_CONFIG:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-10}"
PORT="${PORT:-8020}"
HOST="${HOST:-127.0.0.1}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
SEED="${SEED:-7}"
PROMPT_OVERRIDE_FILE="${PROMPT_OVERRIDE_FILE:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-libero}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-}"
SERVER_VENV="${SERVER_VENV:-.venv}"
LIBERO_VENV="${LIBERO_VENV:-/tmp/${USER:-$(id -un)}/openpi-libero-eval-${SLURM_JOB_ID:-manual}}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT_DIR/.cache/libero-openpi}"
SETUP_ENV="${SETUP_ENV:-never}"
EVAL_NAME="${EVAL_NAME:-}"
SERVER_PID=""

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -e "$path" ]]; then
    echo "[$(timestamp)] Missing ${description}: ${path}" >&2
    exit 1
  fi
}

libero_env_ready() {
  if [[ ! -x "${LIBERO_VENV}/bin/python" ]]; then
    return 1
  fi

  "${LIBERO_VENV}/bin/python" - <<'PY' >/dev/null 2>&1
import importlib

modules = [
    "imageio",
    "libero.libero.envs",
    "matplotlib",
    "libero.libero",
    "mujoco",
    "openpi_client",
    "robosuite",
    "tyro",
    "wandb",
]
for module in modules:
    importlib.import_module(module)
PY
}

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[$(timestamp)] Stopping policy server ${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT TERM INT

if [[ -z "${POLICY_CONFIG}" ]]; then
  echo "[$(timestamp)] POLICY_CONFIG is required." >&2
  exit 1
fi
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "[$(timestamp)] CHECKPOINT_DIR is required." >&2
  exit 1
fi

require_file "${CHECKPOINT_DIR}" "checkpoint directory"
require_file "${SERVER_VENV}/bin/python" "server Python environment"

if [[ -n "${PROMPT_OVERRIDE_FILE}" ]]; then
  require_file "${PROMPT_OVERRIDE_FILE}" "prompt override file"
fi

if [[ -z "${PROMPT_OVERRIDE_FILE}" && "${POLICY_CONFIG}" == *"_logic"* ]]; then
  PROMPT_OVERRIDE_FILE="data/libero/libero_10_logic_descriptions.json"
  require_file "${PROMPT_OVERRIDE_FILE}" "default logic prompt override file"
fi

case "${SETUP_ENV}" in
  auto|always|never)
    ;;
  *)
    echo "[$(timestamp)] SETUP_ENV must be one of: auto, always, never" >&2
    exit 1
    ;;
esac

mkdir -p "${LIBERO_CONFIG_PATH}" logs/slurm
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: $ROOT_DIR/third_party/libero/libero/libero
bddl_files: $ROOT_DIR/third_party/libero/libero/libero/bddl_files
init_states: $ROOT_DIR/third_party/libero/libero/libero/init_files
datasets: $ROOT_DIR/third_party/libero/libero/datasets
assets: $ROOT_DIR/third_party/libero/libero/libero/assets
EOF

if [[ -z "${EVAL_NAME}" ]]; then
  EVAL_NAME="$(basename "${CHECKPOINT_DIR}")_${TASK_SUITE}"
fi
CHECKPOINT_STEP="$(basename "${CHECKPOINT_DIR}")"
TRAIN_RUN_NAME="$(basename "$(dirname "${CHECKPOINT_DIR}")")"
PROMPT_STYLE="default_prompts"
if [[ -n "${PROMPT_OVERRIDE_FILE}" ]]; then
  PROMPT_STYLE="logic_prompts"
fi
DEFAULT_WANDB_TAGS="eval,libero,${TASK_SUITE},${PROMPT_STYLE},policy:${POLICY_CONFIG},train_run:${TRAIN_RUN_NAME},checkpoint:${CHECKPOINT_STEP}"
if [[ -z "${WANDB_TAGS}" ]]; then
  WANDB_TAGS="${DEFAULT_WANDB_TAGS}"
else
  WANDB_TAGS="${WANDB_TAGS},${DEFAULT_WANDB_TAGS}"
fi
if [[ -z "${WANDB_GROUP}" ]]; then
  WANDB_GROUP="${TRAIN_RUN_NAME}"
fi
VIDEO_OUT_PATH="data/libero/evals/${EVAL_NAME}"
RESULTS_OUT_PATH="${VIDEO_OUT_PATH}/results.json"
mkdir -p "${VIDEO_OUT_PATH}"

if [[ "${SETUP_ENV}" == "always" ]] || { [[ "${SETUP_ENV}" == "auto" ]] && ! libero_env_ready; }; then
  echo "[$(timestamp)] Bootstrapping LIBERO eval environment at ${LIBERO_VENV}"
  mkdir -p "$(dirname "${LIBERO_VENV}")"
  if [[ ! -x "${LIBERO_VENV}/bin/python" ]]; then
    uv venv --python 3.10 "${LIBERO_VENV}"
  fi
  # shellcheck disable=SC1090
  source "${LIBERO_VENV}/bin/activate"
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
  uv pip install -r examples/libero/requirements.in --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
  uv pip install -e packages/openpi-client
  uv pip install -e third_party/libero
fi

require_file "${LIBERO_VENV}/bin/python" "LIBERO Python environment"

export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH
export WANDB_MODE

echo "[$(timestamp)] Host: $(hostname)"
echo "[$(timestamp)] Policy config: ${POLICY_CONFIG}"
echo "[$(timestamp)] Checkpoint: ${CHECKPOINT_DIR}"
echo "[$(timestamp)] Task suite: ${TASK_SUITE}"
echo "[$(timestamp)] Trials per task: ${NUM_TRIALS}"
echo "[$(timestamp)] Port: ${PORT}"
echo "[$(timestamp)] Results path: ${RESULTS_OUT_PATH}"
echo "[$(timestamp)] W&B project: ${WANDB_PROJECT}"
echo "[$(timestamp)] W&B group: ${WANDB_GROUP}"
echo "[$(timestamp)] W&B tags: ${WANDB_TAGS}"
if [[ -n "${PROMPT_OVERRIDE_FILE}" ]]; then
  echo "[$(timestamp)] Prompt override file: ${PROMPT_OVERRIDE_FILE}"
fi

"${SERVER_VENV}/bin/python" scripts/serve_policy.py --port="${PORT}" policy:checkpoint --policy.config="${POLICY_CONFIG}" --policy.dir="${CHECKPOINT_DIR}" &
SERVER_PID=$!

SERVER_READY="0"
for _ in $(seq 1 60); do
  if "${SERVER_VENV}/bin/python" - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    sock.connect((host, port))
PY
  then
    SERVER_READY="1"
    break
  fi
  sleep 2
done

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "[$(timestamp)] Policy server exited before eval started." >&2
  wait "${SERVER_PID}" || true
  exit 1
fi

if [[ "${SERVER_READY}" != "1" ]]; then
  echo "[$(timestamp)] Timed out waiting for policy server on ${HOST}:${PORT}." >&2
  exit 1
fi

EVAL_CMD=(
  "${LIBERO_VENV}/bin/python" examples/libero/main.py
  --args.task-suite-name "${TASK_SUITE}"
  --args.host "${HOST}"
  --args.port "${PORT}"
  --args.num-trials-per-task "${NUM_TRIALS}"
  --args.resize-size "${RESIZE_SIZE}"
  --args.replan-steps "${REPLAN_STEPS}"
  --args.seed "${SEED}"
  --args.video-out-path "${VIDEO_OUT_PATH}"
  --args.results-out-path "${RESULTS_OUT_PATH}"
  --args.wandb-enabled
  --args.wandb-project "${WANDB_PROJECT}"
  --args.wandb-name "${EVAL_NAME}"
  --args.wandb-group "${WANDB_GROUP}"
  --args.wandb-tags-csv "${WANDB_TAGS}"
  --args.policy-config "${POLICY_CONFIG}"
  --args.checkpoint-dir "${CHECKPOINT_DIR}"
  --args.train-run-name "${TRAIN_RUN_NAME}"
)

if [[ -n "${PROMPT_OVERRIDE_FILE}" ]]; then
  EVAL_CMD+=(--args.prompt-override-file "${PROMPT_OVERRIDE_FILE}")
fi

PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/packages/openpi-client/src:${ROOT_DIR}/third_party/libero" "${EVAL_CMD[@]}"
