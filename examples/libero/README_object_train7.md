# LIBERO-Object 7/3 Split

This README documents the concrete split and training setup prepared in this repo for fine-tuning `pi0-FAST` on a subset of LIBERO-Object.

## Split

Split file:

`data/libero/object_7train_3test.json`

Train tasks, official LIBERO-Object order indices `0-6`:

1. `pick up the alphabet soup and place it in the basket`
2. `pick up the cream cheese and place it in the basket`
3. `pick up the salad dressing and place it in the basket`
4. `pick up the bbq sauce and place it in the basket`
5. `pick up the ketchup and place it in the basket`
6. `pick up the tomato sauce and place it in the basket`
7. `pick up the butter and place it in the basket`

Held-out test tasks, indices `7-9`:

1. `pick up the milk and place it in the basket`
2. `pick up the chocolate pudding and place it in the basket`
3. `pick up the orange juice and place it in the basket`

## Raw Data

Raw RLDS download location:

`data/libero/raw/libero_object_no_noops`

Download status can be checked with:

```bash
find data/libero/raw/libero_object_no_noops/1.0.0 -maxdepth 1 -type f -name 'libero_object-train.tfrecord-*' | wc -l
```

Expected value after full download: `32`

## Local Training Dataset

Prepared local LeRobot dataset id:

`local/libero_object_train7`

LeRobot stores this at:

`~/.cache/huggingface/lerobot/local/libero_object_train7`

If you need to rebuild the training dataset from the raw RLDS shards:

```bash
PYTHONPATH=src .venv/bin/python examples/libero/convert_libero_data_to_lerobot.py \
  --data_dir data/libero/raw \
  --repo_name local/libero_object_train7 \
  --suite_names libero_object \
  --task_split_file data/libero/object_7train_3test.json \
  --task_split train
```

## Training Configs

Two named configs are prepared for this split:

- `pi0_fast_libero_object_train7`
- `pi0_fast_libero_object_train7_low_mem_finetune`

They both point at:

- dataset repo id: `local/libero_object_train7`
- local-only loading: `true`
- norm-stats asset id: `libero_object_train7`

The first config is full fine-tuning. The second is LoRA / low-memory fine-tuning.

## Normalization Stats

Normalization stats have already been generated for this dataset at:

`assets/pi0_fast_libero_object_train7/libero_object_train7/norm_stats.json`

If you need to recompute them:

```bash
USE_TF=0 uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_object_train7 --num-workers=0
```

The LoRA config uses the same dataset and same stats, so you do not need to recompute them separately.

## Prepared Dataset Summary

The converted local training dataset currently contains:

- `314` training episodes
- `46,286` frames
- `7` train tasks

## Training Commands

Full fine-tune:

```bash
USE_TF=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero_object_train7 \
  --exp-name=object_train7_full \
  --overwrite
```

For a SLURM cluster, use the prepared batch script:

```bash
sbatch examples/libero/train_object_train7_full.slurm
```

That script is configured for full fine-tuning on a single `A100 80GB`-class GPU with:

- `1` GPU
- `16` CPU cores
- `128G` host RAM
- `24:00:00` wall time

It also:

- uses `--resume` so requeued jobs continue from the latest checkpoint
- saves checkpoints every `200` steps
- keeps periodic checkpoints every `1000` steps
- requests requeue on SLURM timeout / preemption signals

You can override the experiment name or worker count at submission time:

```bash
sbatch --export=ALL,EXP_NAME=object_train7_full_cluster,NUM_WORKERS=8 examples/libero/train_object_train7_full.slurm
```

If your cluster uses different GPU labels or requires an account / partition, edit the `#SBATCH` lines in:

`examples/libero/train_object_train7_full.slurm`

LoRA / low-memory fine-tune:

```bash
USE_TF=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero_object_train7_low_mem_finetune \
  --exp-name=object_train7_lora \
  --overwrite
```

## Held-Out Evaluation

For a tmux-based evaluation workflow that serves a checkpoint and runs LIBERO evaluation in one command, see:

`docs/libero_eval_tmux.md`

For the simpler manual two-window workflow focused on the finished LoRA checkpoint, see:

`readme_lora_eval.md`

Serve a checkpoint:

```bash
USE_TF=0 uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi0_fast_libero_object_train7 \
  --policy.dir=checkpoints/pi0_fast_libero_object_train7/object_train7_full/30000
```

Then evaluate only the held-out tasks:

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PWD/third_party/libero
export LIBERO_CONFIG_PATH=$PWD/.cache/libero-openpi
export USE_TF=0

python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-split-file data/libero/object_7train_3test.json \
  --args.task-split eval
```

If you trained with the LoRA config, use `--policy.config=pi0_fast_libero_object_train7_low_mem_finetune` when serving that checkpoint.
