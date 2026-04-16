#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="libero-eval"
PORT="8000"
TASK_SUITE="libero_object"
NUM_TRIALS="50"
HOST="127.0.0.1"
RESIZE_SIZE="224"
REPLAN_STEPS="5"
SEED="7"
VIDEO_OUT_PATH="data/libero/videos"
POLICY_CONFIG="pi0_fast_libero_object_train7_low_mem_finetune"
CHECKPOINT_DIR="checkpoints/pi0_fast_libero_object_train7_low_mem_finetune/object_train7_lora/29999"
LIBERO_VENV="examples/libero/.venv"
SERVER_VENV=".venv"
LIBERO_CONFIG_PATH_DEFAULT="$ROOT_DIR/.cache/libero-openpi"
SETUP_ENV="auto"
ATTACH="1"
EXTRA_EVAL_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/eval_libero_tmux.sh [options] [-- <extra eval args>]

Starts a tmux session with:
  1. a policy server window
  2. a LIBERO eval window

Defaults are tuned for the local LoRA LIBERO object checkpoint.

Options:
  --session NAME           tmux session name (default: libero-eval)
  --port PORT              websocket server port (default: 8000)
  --suite NAME             LIBERO suite name (default: libero_object)
  --trials N               rollouts per task (default: 50)
  --policy-config NAME     training config used to load checkpoint
  --checkpoint-dir PATH    checkpoint directory to evaluate
  --venv PATH              venv path for LIBERO eval (default: examples/libero/.venv)
  --server-venv PATH       venv path for OpenPI server (default: .venv)
  --libero-config-path P   where to write the LIBERO config override
  --setup-env MODE         one of: auto, always, never (default: auto)
  --detach                 create session but do not attach
  --help                   show this message

Examples:
  scripts/eval_libero_tmux.sh
  scripts/eval_libero_tmux.sh --trials 2
  scripts/eval_libero_tmux.sh --session object10 --port 8010
  scripts/eval_libero_tmux.sh -- --args.task-indices 0 1 2
EOF
}

LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --suite)
      TASK_SUITE="$2"
      shift 2
      ;;
    --trials)
      NUM_TRIALS="$2"
      shift 2
      ;;
    --policy-config)
      POLICY_CONFIG="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --venv)
      LIBERO_VENV="$2"
      shift 2
      ;;
    --server-venv)
      SERVER_VENV="$2"
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
    --detach)
      ATTACH="0"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_EVAL_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed." >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi

if [[ ! -x "$SERVER_VENV/bin/python" ]]; then
  echo "OpenPI server environment not found: $SERVER_VENV" >&2
  echo "Create it with 'uv sync' or pass --server-venv PATH." >&2
  exit 1
fi

libero_env_ready() {
  if [[ ! -x "$LIBERO_VENV/bin/python" ]]; then
    return 1
  fi

  "$LIBERO_VENV/bin/python" - <<'PY' >/dev/null 2>&1
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

case "$SETUP_ENV" in
  auto|always|never)
    ;;
  *)
    echo "--setup-env must be one of: auto, always, never" >&2
    exit 1
    ;;
esac

mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $ROOT_DIR/third_party/libero/libero/libero
bddl_files: $ROOT_DIR/third_party/libero/libero/libero/bddl_files
init_states: $ROOT_DIR/third_party/libero/libero/libero/init_files
datasets: $ROOT_DIR/third_party/libero/libero/datasets
assets: $ROOT_DIR/third_party/libero/libero/libero/assets
EOF

SETUP_COMMANDS=()
if [[ "$SETUP_ENV" == "always" ]] || { [[ "$SETUP_ENV" == "auto" ]] && ! libero_env_ready; }; then
  if [[ ! -x "$LIBERO_VENV/bin/python" ]]; then
    SETUP_COMMANDS+=("uv venv --python 3.10 \"$LIBERO_VENV\"")
  fi
  SETUP_COMMANDS+=("source \"$LIBERO_VENV/bin/activate\"")
  SETUP_COMMANDS+=("uv pip install -r examples/libero/requirements.in --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match")
  SETUP_COMMANDS+=("uv pip install -e packages/openpi-client")
  SETUP_COMMANDS+=("uv pip install -e third_party/libero")
fi

SERVER_ENV_COMMANDS=(
  "cd \"$ROOT_DIR\""
  "source \"$SERVER_VENV/bin/activate\""
  "export USE_TF=0"
)

EVAL_ENV_COMMANDS=(
  "cd \"$ROOT_DIR\""
  "source \"$LIBERO_VENV/bin/activate\""
  "export PYTHONPATH=\"$ROOT_DIR/src:$ROOT_DIR/packages/openpi-client/src:$ROOT_DIR/third_party/libero\""
  "export LIBERO_CONFIG_PATH=\"$LIBERO_CONFIG_PATH\""
  "export USE_TF=0"
)

SERVER_COMMANDS=(
  "${SERVER_ENV_COMMANDS[@]}"
  "python scripts/serve_policy.py --port=\"$PORT\" policy:checkpoint --policy.config=\"$POLICY_CONFIG\" --policy.dir=\"$CHECKPOINT_DIR\""
)

EVAL_COMMANDS=(
  "${SETUP_COMMANDS[@]}"
  "${EVAL_ENV_COMMANDS[@]}"
  "python examples/libero/main.py --args.task-suite-name \"$TASK_SUITE\" --args.host \"$HOST\" --args.port \"$PORT\" --args.num-trials-per-task \"$NUM_TRIALS\" --args.resize-size \"$RESIZE_SIZE\" --args.replan-steps \"$REPLAN_STEPS\" --args.seed \"$SEED\" --args.video-out-path \"$VIDEO_OUT_PATH\""
)

for arg in "${EXTRA_EVAL_ARGS[@]}"; do
  EVAL_COMMANDS[-1]+=" $(printf '%q' "$arg")"
done

write_runner_script() {
  local path="$1"
  shift
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -euo pipefail'
    local cmd
    for cmd in "$@"; do
      printf '%s\n' "$cmd"
    done
  } > "$path"
  chmod +x "$path"
}

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "Use a different --session name or kill the existing session first." >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n server
tmux set-option -t "$SESSION_NAME" remain-on-exit on
SERVER_RUNNER="$(mktemp /tmp/openpi-libero-server.XXXXXX.sh)"
EVAL_RUNNER="$(mktemp /tmp/openpi-libero-eval.XXXXXX.sh)"
write_runner_script "$SERVER_RUNNER" "${SERVER_COMMANDS[@]}"
write_runner_script "$EVAL_RUNNER" "sleep 5" "${EVAL_COMMANDS[@]}"

tmux send-keys -t "$SESSION_NAME:server" "bash \"$SERVER_RUNNER\"" C-m

tmux new-window -t "$SESSION_NAME" -n eval
tmux send-keys -t "$SESSION_NAME:eval" "bash \"$EVAL_RUNNER\"" C-m

echo "Created tmux session: $SESSION_NAME"
echo "Server window: ${SESSION_NAME}:server"
echo "Eval window:   ${SESSION_NAME}:eval"
echo "Checkpoint:    $CHECKPOINT_DIR"
echo "Suite:         $TASK_SUITE"
echo "Trials/task:   $NUM_TRIALS"
echo
echo "Attach with:"
echo "  tmux attach -t $SESSION_NAME"

if [[ "$ATTACH" == "1" ]]; then
  exec tmux attach -t "$SESSION_NAME"
fi
