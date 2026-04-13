# LIBERO Object Custom Split Handoff

This document is a self-contained record of the custom `libero_object` split that was prepared and used on this machine, plus the exact steps needed to recreate the same split in a different VLA repo.

The goal is to let you drop this file into another repo and reproduce the same train/eval subset without needing to rediscover the details.

## What We Did

We took the standard 10-task `libero_object` suite and trained on 7 tasks while holding out 3 tasks for evaluation.

The exact 7 training tasks are:

1. `pick up the alphabet soup and place it in the basket`
2. `pick up the cream cheese and place it in the basket`
3. `pick up the salad dressing and place it in the basket`
4. `pick up the bbq sauce and place it in the basket`
5. `pick up the ketchup and place it in the basket`
6. `pick up the tomato sauce and place it in the basket`
7. `pick up the butter and place it in the basket`

The exact 3 held-out evaluation tasks are:

1. `pick up the milk and place it in the basket`
2. `pick up the chocolate pudding and place it in the basket`
3. `pick up the orange juice and place it in the basket`

Official LIBERO task indices used here:

- train: `0 1 2 3 4 5 6`
- eval: `7 8 9`

In short:

- 7-train split: alphabet soup, cream cheese, salad dressing, bbq sauce, ketchup, tomato sauce, butter
- 3-eval split: milk, chocolate pudding, orange juice

## Canonical Split File

In this repo, the split file is:

`data/libero/object_7train_3test.json`

Its contents are:

```json
{
  "suite_name": "libero_object",
  "train_task_indices": [0, 1, 2, 3, 4, 5, 6],
  "eval_task_indices": [7, 8, 9],
  "train_task_instructions": [
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket"
  ],
  "eval_task_instructions": [
    "pick up the milk and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket"
  ]
}
```

If the other repo does not already support a split-file format, copying this JSON and filtering by `train_task_indices` or `train_task_instructions` is enough.

## Machine-Specific Paths From This Computer

Repo root on this machine:

`/home/christopher/Documents/openpi-finetune/openpi`

Raw LIBERO RLDS dataset root used here:

`/home/christopher/Documents/openpi-finetune/openpi/data/libero/raw`

Raw `libero_object` shards on this machine:

`/home/christopher/Documents/openpi-finetune/openpi/data/libero/raw/libero_object_no_noops`

The expected shard count after a complete raw download is:

`32`

Converted local LeRobot dataset id used here:

`local/libero_object_train7`

Where that converted dataset is actually stored on this machine:

`/home/christopher/.cache/huggingface/lerobot/local/libero_object_train7`

Subset metadata written by the converter:

`/home/christopher/.cache/huggingface/lerobot/local/libero_object_train7/meta/libero_subset.json`

That metadata shows:

- selected episode counts:
  - alphabet soup: `44`
  - cream cheese: `45`
  - salad dressing: `47`
  - bbq sauce: `46`
  - ketchup: `45`
  - tomato sauce: `42`
  - butter: `45`
- skipped episode counts:
  - milk: `45`
  - chocolate pudding: `50`
  - orange juice: `45`

Total converted training episodes on this machine:

`314`

Precomputed norm stats in this repo:

`/home/christopher/Documents/openpi-finetune/openpi/assets/pi0_fast_libero_object_train7/libero_object_train7/norm_stats.json`

LIBERO evaluator venv on this machine:

`/home/christopher/Documents/openpi-finetune/openpi/examples/libero/.venv`

That venv currently resolves to:

`/usr/bin/python3.10`

Repo-local LIBERO config dir used for evaluation here:

`/home/christopher/Documents/openpi-finetune/openpi/.cache/libero-openpi`

## Reproducing The Same Split In Another Repo

### 1. Put the split JSON in the new repo

Create a file like `data/libero/object_7train_3test.json` with the JSON above.

If the new repo prefers task names instead of indices, use the task strings exactly as written above. Do not paraphrase them; matching usually depends on exact LIBERO task instructions.

### 2. Download the raw LIBERO RLDS data

The raw dataset source used here is Hugging Face dataset:

`openvla/modified_libero_rlds`

What matters for this split is that the `libero_object_no_noops` raw dataset is available locally.

If you are doing this on a different machine, it is fine to download the vanilla/full LIBERO object dataset from Hugging Face first and then recreate the same 7/3 split yourself locally. You do not need this machine's cached subset as long as you apply the same task split when building the training dataset.

On this machine, it lives under:

`/home/christopher/Documents/openpi-finetune/openpi/data/libero/raw/libero_object_no_noops`

If you want a simple verification check after download:

```bash
find data/libero/raw/libero_object_no_noops/1.0.0 -maxdepth 1 -type f -name 'libero_object-train.tfrecord-*' | wc -l
```

Expected result:

```bash
32
```

### 3. Convert only the 7 training tasks into the new repo's training format

In OpenPI, we converted RLDS to a local LeRobot dataset. The important idea is not LeRobot specifically; the important idea is:

- load the full `libero_object` raw dataset
- keep only tasks `0..6`
- discard tasks `7..9` from the training dataset
- preserve the held-out tasks only for evaluation

The OpenPI conversion command used here was:

```bash
uv run examples/libero/create_task_split.py data/libero/object_7train_3test.json libero_object \
  --train_task_indices 0 1 2 3 4 5 6

uv run examples/libero/convert_libero_data_to_lerobot.py \
  --data_dir data/libero/raw \
  --repo_name local/libero_object_train7 \
  --suite_names libero_object \
  --task_split_file data/libero/object_7train_3test.json \
  --task_split train
```

The resulting local dataset was written to:

`/home/christopher/.cache/huggingface/lerobot/local/libero_object_train7`

If the new repo uses a different dataset format, the equivalent logic is:

1. Read each `libero_object` episode.
2. Inspect its language instruction.
3. Keep it only if the instruction is one of the 7 train tasks above.
4. Build the training dataset from only those retained episodes.

### 4. Point training at the converted subset, not the full 10-task dataset

This is the step that matters most when porting to another VLA repo.

In OpenPI, the prepared training dataset id was:

`local/libero_object_train7`

The OpenPI configs then explicitly used:

- dataset repo id: `local/libero_object_train7`
- local-only dataset loading: `true`
- subset-specific norm stats id: `libero_object_train7`

If the new repo has any notion of:

- dataset name
- data cache path
- normalization stats
- prompt/task metadata

make sure those all point at the 7-task subset, not the full `libero_object` dataset.

Otherwise it is easy to accidentally train on all 10 tasks.

### 5. Recompute normalization stats for the subset if the new repo needs them

OpenPI required separate normalization stats for the subset dataset, and those were stored at:

`/home/christopher/Documents/openpi-finetune/openpi/assets/pi0_fast_libero_object_train7/libero_object_train7/norm_stats.json`

In OpenPI the recompute command was:

```bash
USE_TF=0 uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_object_train7 --num-workers=0
```

For a different repo, the rule is the same:

- if normalization depends on the dataset distribution
- and the dataset is now only the 7-task subset

then recompute stats from that subset instead of reusing stats from the full 10-task `libero_object` dataset.

### 6. Evaluate on only the held-out 3 tasks

Training uses the 7 train tasks.

Held-out evaluation should use only:

- milk
- chocolate pudding
- orange juice

In OpenPI the evaluator can consume the same split JSON and select `eval`:

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-split-file data/libero/object_7train_3test.json \
  --args.task-split eval
```

If the new repo has its own evaluator, mirror the same rule:

- train on indices `0..6`
- evaluate generalization on indices `7..9`

## OpenPI-Specific Details From This Repo

These are not required for another VLA repo, but they document exactly what existed here.

Prepared OpenPI train configs:

- `pi0_fast_libero_object_train7`
- `pi0_fast_libero_object_train7_low_mem_finetune`

Those configs live in:

`src/openpi/training/config.py`

The matching dataset config in this repo points at:

- `repo_id="local/libero_object_train7"`
- `local_files_only=True`
- `asset_id="libero_object_train7"`

Finished checkpoint families referenced in this repo:

- full fine-tune: `checkpoints/pi0_fast_libero_object_train7/object_train7_full`
- LoRA fine-tune: `checkpoints/pi0_fast_libero_object_train7_low_mem_finetune/object_train7_lora`

## Important Environment Notes From This Machine

These mattered during evaluation on this computer:

- OpenPI training and policy serving run from the root repo environment with Python `3.11+`
- the LIBERO evaluator runs from `examples/libero/.venv` with Python `3.10`
- the evaluator uses `LIBERO_CONFIG_PATH=/home/christopher/Documents/openpi-finetune/openpi/.cache/libero-openpi`

The repo-local LIBERO config was necessary because the global `~/.libero/config.yaml` on this machine pointed at a different LIBERO checkout.

If the new repo also vendors LIBERO or uses repo-local assets, do not assume the global LIBERO config is correct.

## Minimal Porting Checklist

When you move this workflow to another VLA repo, verify all of the following:

1. The split JSON exists and exactly matches the 7 train / 3 eval tasks above.
2. The training dataset contains only the 7 train tasks.
3. The training code points at the subset dataset, not the full `libero_object` dataset.
4. Any normalization or dataset statistics are recomputed from the subset if needed.
5. Held-out evaluation runs only on milk, chocolate pudding, and orange juice.
6. Any cached dataset path used by the new repo is documented explicitly, the same way it is documented here.

## Short Version

If you only need the essential facts to recreate this in another repo:

- suite: `libero_object`
- train task indices: `0 1 2 3 4 5 6`
- eval task indices: `7 8 9`
- raw data source used here: `openvla/modified_libero_rlds`
- raw local path on this machine: `/home/christopher/Documents/openpi-finetune/openpi/data/libero/raw/libero_object_no_noops`
- converted local dataset name used here: `local/libero_object_train7`
- converted dataset path on this machine: `/home/christopher/.cache/huggingface/lerobot/local/libero_object_train7`
- total converted train episodes here: `314`
- held-out tasks: milk, chocolate pudding, orange juice
