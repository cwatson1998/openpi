# `brandonyang/combined_chris_robot_dataset` notes

Dataset snapshot:

- repo id: `brandonyang/combined_chris_robot_dataset`
- local download target: `data/hf/combined_chris_robot_dataset`
- downloader log: `logs/combined_chris_robot_dataset_download.log`
- downloader pid file: `logs/combined_chris_robot_dataset_download.pid`
- approximate remote size: `11,844,526,723` bytes (`~11.8 GB`)

What the metadata says:

- format: already a LeRobot dataset, so this is much closer to trainable than raw teleop logs
- robot type: `panda`
- total episodes: `224`
- total frames: `51,656`
- total tasks: `15`
- fps: `15`
- videos: `0`
- split: `train` only
- episode length range: `138` to `396` frames
- mean episode length: `230.61` frames

Observed feature schema:

- images:
  - `exterior_image_1_left`
  - `exterior_image_2_left`
  - `wrist_image_left`
- proprio:
  - `joint_position` with shape `[7]`
  - `gripper_position` with shape `[1]`
- action:
  - `actions` with shape `[8]`

Task distribution:

- 14: `Move the chicken to the right.`
- 14: `Move the coffee pod to the left.`
- 15: `Pick the pineapple from the bowl and place it into the basket.`
- 15: `Place the marker inside the bowl.`
- 13: `Stack the red block on top of the green block.`
- 14: `Move the banana to the left.`
- 14: `Move the chicken to the left of the cup.`
- 15: `Move the giraffe to the right.`
- 14: `Move the green block to the right of the black block.`
- 14: `Move the pineapple to the left of the banana.`
- 14: `Move the sprinkles towards the Coke.`
- 15: `Move the watermelon to the right of the cloth.`
- 20: `Move the banana towards the tennis ball.`
- 14: `Take the item off the top of the book and place it on the left side of the bowl.`
- 19: `Move the Skittles away from the bowl.`

What to explore next before training:

1. Confirm action semantics.
   The dataset has `actions` with dimension `8`, while the Libero example uses `7`. For Panda this probably means `7` arm joints plus `1` gripper command, but you should verify whether those are absolute targets, deltas, or velocities. This directly determines whether the `DeltaActions` transform from the Libero config should stay, be removed, or be changed.

2. Map dataset keys to OpenPI policy inputs.
   The existing Libero config expects dataset keys like `observation/image`, `observation/wrist_image`, `observation/state`, and `actions`. Your dataset instead uses `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left`, `joint_position`, and `gripper_position`. You will need a custom `DataConfigFactory` and likely custom policy input transforms.

3. Decide which cameras to use.
   You have three image streams and no videos. OpenPI can work with missing views, but you should choose whether to train with one exterior view plus wrist, both exterior views plus wrist, or some ablation. The note in `src/openpi/policies/libero_policy.py` is the place to mirror.

4. Build the state vector deliberately.
   The dataset stores `joint_position` and `gripper_position` separately. Decide whether inference on your Panda setup can provide the same state vector in the same order and units. Training and inference keys must match.

5. Inspect prompt quality.
   The dataset already has 15 natural-language tasks, which is good. Check whether each episode carries a single clean instruction and whether phrasing is stable enough for generalization.

6. Check for train/eval splitting.
   The dataset currently exposes only a `train` split. For a real-world fine-tune, create your own held-out split by task and/or by episode so you can measure overfitting.

7. Check normalization stats once the local dataset is wired into a config.
   After you make a custom config, run `scripts/compute_norm_stats.py` against that config before training.

Likely repo changes if you want to fine-tune this dataset:

1. Copy `LeRobotLiberoDataConfig` in `src/openpi/training/config.py` and make a Panda-real variant that repacks your dataset keys.
2. Copy or adapt the Libero policy transforms so input images and state map correctly for both training and inference.
3. Change `pi0_fast` model config to `action_dim=8` unless inspection shows one action should be dropped or transformed.
4. Decide whether `action_horizon=10` is still appropriate for your teleop data.
5. Add a new `TrainConfig` pointing at `brandonyang/combined_chris_robot_dataset` or at the fully downloaded local copy with `local_files_only=True`.

How to check whether the background download finished:

Run these from the repo root:

```bash
ps -fp "$(cat logs/combined_chris_robot_dataset_download.pid)"
tail -f logs/combined_chris_robot_dataset_download.log
find data/hf/combined_chris_robot_dataset -type f | wc -l
du -sh data/hf/combined_chris_robot_dataset
```

Practical finish signals:

- the process in the pid file no longer exists
- the log ends with `Download complete: ...`
- the local directory contains `224` parquet files plus the `meta/*` files
- the directory size is roughly `11.8G`
