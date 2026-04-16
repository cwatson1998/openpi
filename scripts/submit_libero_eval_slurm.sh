#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POLICY_CONFIG=""
CHECKPOINT_DIR=""
TASK_SUITE="libero_10"
NUM_TRIALS="10"
PORT=""
CPUS="8"
MEMORY="64G"
TIME_LIMIT="06:00:00"
SETUP_ENV="auto"
PROMPT_OVERRIDE_FILE=""
WANDB_PROJECT="libero"
WANDB_GROUP=""
WANDB_TAGS=""
SERVER_VENV=".venv"
LIBERO_VENV=""
NODELIST="dj-a40-0.grasp.maas,dj-a40-1.grasp.maas,dj-l40-0.grasp.maas"
EVAL_NAME=""

usage() {
  cat <<'EOF'
Usage:
  scripts/submit_libero_eval_slurm.sh [options]

Submits a single-GPU LIBERO checkpoint eval job to Slurm. Defaults are tuned for
24G+ VRAM nodes. If --policy-config is omitted, the script tries to infer it from
checkpoints/<config>/... in --checkpoint-dir.

Options:
  --policy-config NAME        training config that matches the checkpoint
  --checkpoint-dir PATH       checkpoint directory to evaluate
  --suite NAME                LIBERO suite name (default: libero_10)
  --trials N                  rollouts per task (default: 10)
  --prompt-override-file P    optional prompt override JSON for logic prompts
  --name NAME                 eval run name used under data/libero/evals/
  --wandb-project NAME        W&B project for eval logging (default: libero)
  --wandb-group NAME          optional W&B group override
  --wandb-tags CSV            optional extra comma-separated W&B tags
  --port PORT                 policy server port inside the job (default: auto-derived)
  --cpus N                    cpus per task (default: 8)
  --mem STR                   memory request (default: 64G)
  --time HH:MM:SS             wall time (default: 06:00:00)
  --nodelist CSV              comma-separated safe nodes
  --setup-env MODE            auto, always, or never (default: auto)
  --server-venv PATH          OpenPI server venv (default: .venv)
  --libero-venv PATH          LIBERO eval venv override (default: job-local /tmp env)
  --help                      show this message

Examples:
  scripts/submit_libero_eval_slurm.sh \
    --policy-config pi0_fast_libero_10_logic \
    --checkpoint-dir checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_20260414_174057/23600 \
    --prompt-override-file data/libero/libero_10_logic_descriptions.json

  scripts/submit_libero_eval_slurm.sh \
    --policy-config pi0_fast_libero \
    --checkpoint-dir checkpoints/pi0_fast_libero/libero10_default_ctrl_20260414_174400/200
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-config)
      POLICY_CONFIG="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
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
    --prompt-override-file)
      PROMPT_OVERRIDE_FILE="$2"
      shift 2
      ;;
    --name)
      EVAL_NAME="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-group)
      WANDB_GROUP="$2"
      shift 2
      ;;
    --wandb-tags)
      WANDB_TAGS="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --mem)
      MEMORY="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --nodelist)
      NODELIST="$2"
      shift 2
      ;;
    --setup-env)
      SETUP_ENV="$2"
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

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "--checkpoint-dir is required." >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if [[ -n "${PROMPT_OVERRIDE_FILE}" && ! -f "${PROMPT_OVERRIDE_FILE}" ]]; then
  echo "Prompt override file not found: ${PROMPT_OVERRIDE_FILE}" >&2
  exit 1
fi

if [[ -z "${POLICY_CONFIG}" ]]; then
  checkpoint_abs="$(realpath "${CHECKPOINT_DIR}")"
  if [[ "${checkpoint_abs}" == *"/checkpoints/"* ]]; then
    prefix="${checkpoint_abs#*"/checkpoints/"}"
    POLICY_CONFIG="${prefix%%/*}"
  fi
fi

if [[ -z "${POLICY_CONFIG}" ]]; then
  echo "--policy-config is required when it cannot be inferred from checkpoints/<config>/..." >&2
  exit 1
fi

if [[ -z "${PROMPT_OVERRIDE_FILE}" && "${POLICY_CONFIG}" == *"_logic"* ]]; then
  PROMPT_OVERRIDE_FILE="data/libero/libero_10_logic_descriptions.json"
fi

if [[ -n "${PROMPT_OVERRIDE_FILE}" && ! -f "${PROMPT_OVERRIDE_FILE}" ]]; then
  echo "Prompt override file not found: ${PROMPT_OVERRIDE_FILE}" >&2
  exit 1
fi

if [[ -z "${EVAL_NAME}" ]]; then
  checkpoint_name="$(basename "${CHECKPOINT_DIR}")"
  parent_name="$(basename "$(dirname "${CHECKPOINT_DIR}")")"
  EVAL_NAME="${parent_name}_${checkpoint_name}_${TASK_SUITE}"
fi

if [[ -z "${PORT}" ]]; then
  port_seed="$(printf '%s' "${EVAL_NAME}" | cksum | awk '{print $1}')"
  PORT="$((8200 + (port_seed % 600)))"
fi

EXPORTS=(
  "POLICY_CONFIG=${POLICY_CONFIG}"
  "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
  "TASK_SUITE=${TASK_SUITE}"
  "NUM_TRIALS=${NUM_TRIALS}"
  "PORT=${PORT}"
  "SETUP_ENV=${SETUP_ENV}"
  "WANDB_PROJECT=${WANDB_PROJECT}"
  "SERVER_VENV=${SERVER_VENV}"
  "EVAL_NAME=${EVAL_NAME}"
)

if [[ -n "${PROMPT_OVERRIDE_FILE}" ]]; then
  EXPORTS+=("PROMPT_OVERRIDE_FILE=${PROMPT_OVERRIDE_FILE}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
  EXPORTS+=("WANDB_GROUP=${WANDB_GROUP}")
fi
if [[ -n "${WANDB_TAGS}" ]]; then
  EXPORTS+=("WANDB_TAGS=${WANDB_TAGS}")
fi
if [[ -n "${LIBERO_VENV}" ]]; then
  EXPORTS+=("LIBERO_VENV=${LIBERO_VENV}")
fi

printf 'Submitting LIBERO eval:\n'
printf '  policy-config: %s\n' "${POLICY_CONFIG}"
printf '  checkpoint:    %s\n' "${CHECKPOINT_DIR}"
printf '  suite:         %s\n' "${TASK_SUITE}"
printf '  trials/task:   %s\n' "${NUM_TRIALS}"
printf '  nodelist:      %s\n' "${NODELIST}"
printf '  eval name:     %s\n' "${EVAL_NAME}"
printf '  wandb project: %s\n' "${WANDB_PROJECT}"

sbatch \
  --partition=dineshj-compute \
  --qos=dj-med \
  --cpus-per-task="${CPUS}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  --gres=gpu:1 \
  --nodelist="${NODELIST}" \
  --export=ALL,"$(IFS=,; echo "${EXPORTS[*]}")" \
  examples/libero/eval_checkpoint.slurm
