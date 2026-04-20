#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POLICY_CONFIG="pi0_fast_libero_10_logic_low_mem_finetune"
TASK_SUITE="libero_10"
NUM_TRIALS="10"
HOST="127.0.0.1"
PORT="8000"
RESIZE_SIZE="224"
REPLAN_STEPS="5"
SEED="7"
SERVER_VENV=".venv"
LIBERO_VENV="examples/libero/.venv"
LIBERO_CONFIG_PATH="$ROOT_DIR/.cache/libero-openpi"
PROMPT_OVERRIDE_FILE="data/libero/libero_10_logic_descriptions.json"
SETUP_ENV="auto"
WANDB_PROJECT="libero"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_libero10_logic_lora_pair}"
OUTPUT_ROOT="data/libero/evals/${RUN_ID}"
LOG_ROOT="logs/libero_eval/${RUN_ID}"
ONLY_CHECKPOINT=""

CHECKPOINT_NAMES=(
  "lora"
  "lora-from-libero-ckpt"
)
CHECKPOINT_DIRS=(
  "checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_20260416_013843/29999"
  "checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_from_libero_ckpt_20260416_013843/29999"
)

SERVER_PID=""

usage() {
  cat <<'EOF'
Usage:
  scripts/eval_libero10_logic_lora_pair.sh [options]

Evaluates the two local LIBERO-10 logic LoRA checkpoints sequentially:
  1. starts the policy server for checkpoint A
  2. waits for the websocket port to accept connections
  3. runs 10 LIBERO-10 rollouts per task
  4. tears the server down
  5. repeats for checkpoint B

Options:
  --port PORT              websocket port for the local policy server (default: 8000)
  --server-venv PATH       OpenPI server venv (default: .venv)
  --libero-venv PATH       LIBERO eval venv (default: examples/libero/.venv)
  --libero-config-path P   repo-local LIBERO config dir (default: .cache/libero-openpi)
  --setup-env MODE         one of: auto, always, never (default: auto)
  --wandb-project NAME     W&B project name (default: libero)
  --run-id NAME            output/log namespace for this pair run
  --only NAME              evaluate only one checkpoint short name
  --help                   show this message
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "Stopping policy server pid=${SERVER_PID}"
    kill -INT "${SERVER_PID}" 2>/dev/null || true

    local waited=0
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
      if (( waited >= 15 )); then
        log "Policy server did not exit after SIGINT; sending SIGTERM"
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
      fi
      if (( waited >= 25 )); then
        log "Policy server still alive; sending SIGKILL"
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
      fi
      ((waited += 1))
      sleep 1
    done

    wait "${SERVER_PID}" || true
  fi
  SERVER_PID=""
}

trap cleanup_server EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --server-venv)
      SERVER_VENV="$2"
      shift 2
      ;;
    --libero-venv)
      LIBERO_VENV="$2"
      shift 2
      ;;
    --libero-config-path)
      LIBERO_CONFIG_PATH="$2"
      shift 2
      ;;
    --setup-env)
      SETUP_ENV="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --only)
      ONLY_CHECKPOINT="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      OUTPUT_ROOT="data/libero/evals/${RUN_ID}"
      LOG_ROOT="logs/libero_eval/${RUN_ID}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$SETUP_ENV" in
  auto|always|never)
    ;;
  *)
    echo "--setup-env must be one of: auto, always, never" >&2
    exit 1
    ;;
esac

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing ${description}: ${path}" >&2
    exit 1
  fi
}

libero_env_ready() {
  if [[ ! -x "${LIBERO_VENV}/bin/python" ]]; then
    return 1
  fi

  PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/packages/openpi-client/src:${ROOT_DIR}/third_party/libero" \
    "${LIBERO_VENV}/bin/python" - <<'PY' >/dev/null 2>&1
import importlib

modules = [
    "imageio",
    "libero.libero",
    "libero.libero.envs",
    "matplotlib",
    "mujoco",
    "openpi_client",
    "pkg_resources",
    "robosuite",
    "tyro",
    "wandb",
]
for module in modules:
    importlib.import_module(module)
PY
}

bootstrap_libero_env() {
  if [[ "$SETUP_ENV" == "never" ]]; then
    return
  fi

  if [[ "$SETUP_ENV" == "always" ]] || ! libero_env_ready; then
    log "Bootstrapping/reparing LIBERO eval environment at ${LIBERO_VENV}"
    mkdir -p "$(dirname "${LIBERO_VENV}")"
    if [[ ! -x "${LIBERO_VENV}/bin/python" ]]; then
      uv venv --python 3.10 "${LIBERO_VENV}"
    fi

    uv pip install --python "${LIBERO_VENV}/bin/python" -r examples/libero/requirements.in \
      --extra-index-url https://download.pytorch.org/whl/cu113 \
      --index-strategy=unsafe-best-match
    uv pip install --python "${LIBERO_VENV}/bin/python" -e packages/openpi-client
    uv pip install --python "${LIBERO_VENV}/bin/python" -e third_party/libero

    # Newer setuptools drops pkg_resources, but wandb still imports it.
    "${LIBERO_VENV}/bin/python" -m ensurepip --upgrade >/dev/null
    "${LIBERO_VENV}/bin/python" -m pip install --upgrade 'setuptools<81'
  fi

  if ! libero_env_ready; then
    echo "LIBERO eval environment is still not ready: ${LIBERO_VENV}" >&2
    exit 1
  fi
}

write_libero_config() {
  mkdir -p "${LIBERO_CONFIG_PATH}"
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${ROOT_DIR}/third_party/libero/libero/libero
bddl_files: ${ROOT_DIR}/third_party/libero/libero/libero/bddl_files
init_states: ${ROOT_DIR}/third_party/libero/libero/libero/init_files
datasets: ${ROOT_DIR}/third_party/libero/libero/datasets
assets: ${ROOT_DIR}/third_party/libero/libero/libero/assets
EOF
}

wait_for_server() {
  local host="$1"
  local port="$2"

  for _ in $(seq 1 60); do
    if "${SERVER_VENV}/bin/python" - "${host}" "${port}" 2>/dev/null <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    sock.connect((host, port))
PY
    then
      return 0
    fi
    sleep 2
  done

  return 1
}

run_checkpoint_eval() {
  local short_name="$1"
  local checkpoint_dir="$2"

  local train_run_name
  train_run_name="$(basename "$(dirname "${checkpoint_dir}")")"
  local ckpt_step
  ckpt_step="$(basename "${checkpoint_dir}")"
  local eval_name="${RUN_ID}_${short_name}"
  local eval_output_dir="${OUTPUT_ROOT}/${short_name}"
  local results_out_path="${eval_output_dir}/results.json"
  local progress_out_path="${eval_output_dir}/eval_progress.json"
  local server_log="${LOG_ROOT}/${short_name}_server.log"
  local eval_log="${LOG_ROOT}/${short_name}_eval.log"
  local wandb_tags
  wandb_tags="eval,libero,libero_10,logic_prompts,local_sequence,policy:${POLICY_CONFIG},train_run:${train_run_name},checkpoint:${ckpt_step}"

  mkdir -p "${eval_output_dir}" "${LOG_ROOT}"

  log "Starting ${short_name}"
  log "Checkpoint: ${checkpoint_dir}"
  log "Output dir: ${eval_output_dir}"
  log "Server log: ${server_log}"
  log "Eval log: ${eval_log}"

  USE_TF=0 TOKENIZERS_PARALLELISM=false LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" WANDB_MODE="${WANDB_MODE}" \
    "${SERVER_VENV}/bin/python" scripts/serve_policy.py \
      --port="${PORT}" \
      policy:checkpoint \
      --policy.config "${POLICY_CONFIG}" \
      --policy.dir "${checkpoint_dir}" \
      >"${server_log}" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server "${HOST}" "${PORT}"; then
    log "Timed out waiting for policy server on ${HOST}:${PORT}"
    if [[ -f "${server_log}" ]]; then
      tail -n 80 "${server_log}" >&2 || true
    fi
    exit 1
  fi

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "Policy server exited before eval started"
    if [[ -f "${server_log}" ]]; then
      tail -n 80 "${server_log}" >&2 || true
    fi
    wait "${SERVER_PID}" || true
    exit 1
  fi

  log "Policy server ready on ${HOST}:${PORT}"

  (
    export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/packages/openpi-client/src:${ROOT_DIR}/third_party/libero"
    export LIBERO_CONFIG_PATH
    export USE_TF=0
    export TOKENIZERS_PARALLELISM=false
    export WANDB_MODE

    "${LIBERO_VENV}/bin/python" examples/libero/main.py \
      --args.task-suite-name "${TASK_SUITE}" \
      --args.host "${HOST}" \
      --args.port "${PORT}" \
      --args.num-trials-per-task "${NUM_TRIALS}" \
      --args.resize-size "${RESIZE_SIZE}" \
      --args.replan-steps "${REPLAN_STEPS}" \
      --args.seed "${SEED}" \
      --args.video-out-path "${eval_output_dir}" \
      --args.results-out-path "${results_out_path}" \
      --args.progress-out-path "${progress_out_path}" \
      --args.prompt-override-file "${PROMPT_OVERRIDE_FILE}" \
      --args.wandb-enabled \
      --args.wandb-project "${WANDB_PROJECT}" \
      --args.wandb-name "${eval_name}" \
      --args.wandb-group "${train_run_name}" \
      --args.wandb-tags-csv "${wandb_tags}" \
      --args.policy-config "${POLICY_CONFIG}" \
      --args.checkpoint-dir "${checkpoint_dir}" \
      --args.train-run-name "${train_run_name}"
  ) 2>&1 | tee "${eval_log}"

  log "Completed ${short_name}"
  cleanup_server
}

main() {
  require_file "${SERVER_VENV}/bin/python" "server Python environment"
  require_file "scripts/serve_policy.py" "policy server entrypoint"
  require_file "examples/libero/main.py" "LIBERO evaluator entrypoint"
  require_file "${PROMPT_OVERRIDE_FILE}" "logic prompt override file"
  require_file "third_party/libero" "third_party/libero checkout"

  for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    require_file "${checkpoint_dir}" "checkpoint directory"
  done

  mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
  write_libero_config
  bootstrap_libero_env

  if [[ -n "${ONLY_CHECKPOINT}" ]]; then
    case "${ONLY_CHECKPOINT}" in
      lora|lora-from-libero-ckpt)
        ;;
      *)
        echo "--only must be one of: lora, lora-from-libero-ckpt" >&2
        exit 1
        ;;
    esac
  fi

  log "Run id: ${RUN_ID}"
  log "Task suite: ${TASK_SUITE}"
  log "Trials per task: ${NUM_TRIALS}"
  log "Output root: ${OUTPUT_ROOT}"
  log "Log root: ${LOG_ROOT}"
  log "W&B project: ${WANDB_PROJECT}"
  log "W&B mode: ${WANDB_MODE}"

  local i
  for i in "${!CHECKPOINT_NAMES[@]}"; do
    if [[ -n "${ONLY_CHECKPOINT}" ]] && [[ "${CHECKPOINT_NAMES[$i]}" != "${ONLY_CHECKPOINT}" ]]; then
      continue
    fi
    run_checkpoint_eval "${CHECKPOINT_NAMES[$i]}" "${CHECKPOINT_DIRS[$i]}"
  done

  log "All checkpoint evals finished"
}

main "$@"
