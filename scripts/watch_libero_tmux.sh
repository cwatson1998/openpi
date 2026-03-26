#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME="libero-eval"
EVAL_WINDOW="eval"
SERVER_WINDOW="server"
INTERVAL="10"
LINES="300"
FOLLOW="1"

usage() {
  cat <<'EOF'
Usage:
  scripts/watch_libero_tmux.sh [options]

Summarizes the current LIBERO tmux evaluation session by inspecting pane output.

Options:
  --session NAME         tmux session name (default: libero-eval)
  --eval-window NAME     eval window name (default: eval)
  --server-window NAME   server window name (default: server)
  --interval SEC         refresh interval in seconds (default: 10)
  --lines N              lines to capture from each pane (default: 300)
  --once                 print one snapshot and exit
  --help                 show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION_NAME="$2"
      shift 2
      ;;
    --eval-window)
      EVAL_WINDOW="$2"
      shift 2
      ;;
    --server-window)
      SERVER_WINDOW="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --lines)
      LINES="$2"
      shift 2
      ;;
    --once)
      FOLLOW="0"
      shift
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

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed." >&2
  exit 1
fi

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session not found: $SESSION_NAME" >&2
  exit 1
fi

print_snapshot() {
  local eval_pane="${SESSION_NAME}:${EVAL_WINDOW}"
  local server_pane="${SESSION_NAME}:${SERVER_WINDOW}"
  local eval_output
  local server_output
  local task_line
  local episodes_line
  local successes_line
  local task_rate_line
  local total_rate_line
  local final_rate_line
  local total_episodes_line
  local server_tail

  eval_output="$(tmux capture-pane -pt "$eval_pane" -S "-$LINES" 2>/dev/null || true)"
  server_output="$(tmux capture-pane -pt "$server_pane" -S "-40" 2>/dev/null || true)"

  task_line="$(printf '%s\n' "$eval_output" | grep -E 'Task: ' | tail -n 1 || true)"
  episodes_line="$(printf '%s\n' "$eval_output" | grep -E '# episodes completed so far:' | tail -n 1 || true)"
  successes_line="$(printf '%s\n' "$eval_output" | grep -E '# successes:' | tail -n 1 || true)"
  task_rate_line="$(printf '%s\n' "$eval_output" | grep -E 'Current task success rate:' | tail -n 1 || true)"
  total_rate_line="$(printf '%s\n' "$eval_output" | grep -E 'Current total success rate:' | tail -n 1 || true)"
  final_rate_line="$(printf '%s\n' "$eval_output" | grep -E '^INFO:root:Total success rate:|^Total success rate:' | tail -n 1 || true)"
  total_episodes_line="$(printf '%s\n' "$eval_output" | grep -E '^INFO:root:Total episodes:|^Total episodes:' | tail -n 1 || true)"
  server_tail="$(printf '%s\n' "$server_output" | tail -n 8)"

  clear
  printf 'LIBERO Eval Watch\n'
  printf 'Session: %s\n' "$SESSION_NAME"
  printf 'Eval window: %s   Server window: %s\n' "$EVAL_WINDOW" "$SERVER_WINDOW"
  printf 'Updated: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf '\n'

  printf 'Eval status\n'
  printf '-----------\n'
  if [[ -n "$task_line" ]]; then
    printf '%s\n' "$task_line"
  else
    printf 'No task line observed yet.\n'
  fi
  if [[ -n "$episodes_line" ]]; then
    printf '%s\n' "$episodes_line"
  fi
  if [[ -n "$successes_line" ]]; then
    printf '%s\n' "$successes_line"
  fi
  if [[ -n "$task_rate_line" ]]; then
    printf '%s\n' "$task_rate_line"
  fi
  if [[ -n "$total_rate_line" ]]; then
    printf '%s\n' "$total_rate_line"
  fi
  if [[ -n "$final_rate_line" ]]; then
    printf '%s\n' "$final_rate_line"
  fi
  if [[ -n "$total_episodes_line" ]]; then
    printf '%s\n' "$total_episodes_line"
  fi
  printf '\n'

  printf 'Server tail\n'
  printf '-----------\n'
  if [[ -n "$server_tail" ]]; then
    printf '%s\n' "$server_tail"
  else
    printf 'No server output captured.\n'
  fi
  printf '\n'

  printf 'Recent eval tail\n'
  printf '----------------\n'
  printf '%s\n' "$eval_output" | tail -n 20
}

if [[ "$FOLLOW" == "0" ]]; then
  print_snapshot
  exit 0
fi

while true; do
  print_snapshot
  sleep "$INTERVAL"
done
