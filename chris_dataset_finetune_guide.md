# Fine-Tuning Pi0.5 on `combined_chris_robot_dataset`

Dataset home:
- https://huggingface.co/datasets/brandonyang/combined_chris_robot_dataset

## Recommendation

This dataset should be treated as a DROID-style Franka dataset, not as a LIBERO dataset.

Why:
- robot type is `panda`
- control frequency is `15 Hz`
- actions are 8-D
- observations include DROID-style keys:
  - `exterior_image_1_left`
  - `exterior_image_2_left`
  - `wrist_image_left`
  - `joint_position`
  - `gripper_position`
- you confirmed the actions are joint velocities, which matches the repo’s documented DROID convention

That means the most sensible starting point is to fine-tune `pi0_fast` using the existing DROID input/output conventions, rather than adapting the LIBERO pipeline.

## Why This Is Plausible

The repo already has a DROID-flavored Pi0.5 config:
- `pi0_fast_droid` in `src/openpi/training/config.py`

The repo also already defines DROID transforms:
- `src/openpi/policies/droid_policy.py`

Those transforms already expect:
- one exterior camera
- one wrist camera
- 7 joint positions + 1 gripper position as state
- 8-D actions

Your dataset is very close to that schema. The main work is wiring your LeRobot dataset into a new training config.

## Suggested Training Design

Use a new config modeled after `pi0_fast_droid`, but pointed at your dataset.

Recommended choices:
- model: `pi0_fast`
- action dimension: `8`
- action horizon: start with `10`
- prompts: use dataset task text from `tasks.jsonl`
- normalization stats: first try DROID stats (`asset_id="droid"`), then compare against freshly computed stats

For cameras:
- start with `exterior_image_1_left` + `wrist_image_left`
- ignore `exterior_image_2_left` in the first pass

Reason:
- this matches the existing DROID policy transform with the least code churn
- it reduces the chance of introducing a camera-mapping bug early

## Proposed Workflow

1. Make the dataset visible to LeRobot as a local dataset.

The current training loader expects datasets under the LeRobot cache layout, for example:
- `~/.cache/huggingface/lerobot/local/libero_object_train7`

So for local-only training, the cleanest path is to place the dataset under something like:
- `~/.cache/huggingface/lerobot/local/combined_chris_robot_dataset`

2. Add a new training config.

The config should:
- use `pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, max_token_len=180)`
- use DROID-style transforms
- point `repo_id` at the local dataset
- set `prompt_from_task=True`
- set `local_files_only=True`
- start with `AssetsConfig(asset_id="droid")`

3. Compute normalization stats for the dataset.

Run:

```bash
USE_TF=0 uv run scripts/compute_norm_stats.py --config-name <your_config_name> --num-workers=0
```

4. Fine-tune.

Start with LoRA / low-memory fine-tuning first, then consider full fine-tuning only if needed.

Example shape of the command:

```bash
cd /path/to/openpi
export USE_TF=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

uv run scripts/train.py \
  <your_config_name> \
  --exp-name=combined_chris_robot_dataset_lora \
  --batch-size=16
```

## Key Risks To Check Early

1. Action semantics

You stated the dataset uses joint velocities. That is good and strongly supports the DROID route. Still verify:
- first 7 dimensions are joint velocities
- 8th dimension is gripper action
- units and gripper convention match repo expectations closely enough

2. Dataset location

The current loader resolves LeRobot datasets by `repo_id`, not arbitrary file path. If the dataset only exists at:
- `data/hf/combined_chris_robot_dataset`

then training will not find it automatically. Put it under the LeRobot local dataset root or add a small loader extension.

3. Camera usage

The existing DROID transform uses one exterior camera plus wrist. Your dataset has two exterior cameras. Ignore the second one initially unless there is a strong reason to use it.

4. Evaluation split

The dataset currently exposes only a training split. Create your own held-out split by task and/or episode if you want meaningful validation.

## Recommended First Experiment

Start with the simplest path:
- single exterior camera: `exterior_image_1_left`
- wrist camera: `wrist_image_left`
- DROID transforms
- `pi0_fast`
- LoRA fine-tuning
- DROID normalization stats first

This gives you the highest chance of getting a usable baseline quickly. Once that works, you can explore:
- custom prompt variants
- using the second exterior camera
- fresh normalization stats
- full fine-tuning
