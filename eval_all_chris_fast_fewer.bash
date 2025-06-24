#!/usr/bin/env bash
set -euo pipefail

# Base paths
CONFIG_DIR="/app/VLABench/VLABench/configs/evaluation/tracks"
SAVE_BASE="${HOME}/data/vlabench_results/pi0_fast_primitive_vis_fewer_two"
SCRIPT="examples/vlabench/eval.py"
TASKS="select_drink select_fruit select_toy"
N_EPISODE=2


# Loop through each JSON config in the tracks directory
for cfg in "$CONFIG_DIR"/*.json; do
  # Get the base name without extension (e.g., "track_1_in_distribution")
  name=$(basename "$cfg" .json)

  # Construct save directory path
  save_dir="$SAVE_BASE/$name"

  # Create the directory if it doesn't exist
  mkdir -p "$save_dir"

  echo "Running eval for config: $cfg"
  echo "   ➤ Saving results to: $save_dir"

  # Run the evaluation
  python "$SCRIPT" \
    --args.episode-config-path "$cfg" \
    --args.save_dir "$save_dir" \
    --args.tasks "$TASKS" \
    --args.n_episode "$N_EPISODE" \
    --args.visulization
done
