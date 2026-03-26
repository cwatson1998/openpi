# LoRA Checkpoint Evaluation

This document describes the simplest manual workflow for evaluating the finished LIBERO LoRA checkpoint from this repo using two terminal windows:

- one window for the policy server
- one window for LIBERO evaluation

This avoids tmux and keeps the control flow explicit.

## Checkpoint

The finished LoRA checkpoint is:

- `checkpoints/pi0_fast_libero_object_train7_low_mem_finetune/object_train7_lora/29999`

The matching training config is:

- `pi0_fast_libero_object_train7_low_mem_finetune`

Those two must stay in sync. If you point at a different checkpoint family, update the config name too.

## Why `LIBERO_CONFIG_PATH` Must Be Overridden

On this machine, the global LIBERO config at `~/.libero/config.yaml` points at a different LIBERO checkout.

If you do not override `LIBERO_CONFIG_PATH`, the evaluator may load:

- assets from the wrong repo
- BDDL files from the wrong repo
- init states from the wrong repo

So for this repo, use a repo-local LIBERO config directory instead of relying on `~/.libero`.

This README uses:

- config dir: `.cache/libero-openpi`
- config file: `.cache/libero-openpi/config.yaml`

## One-Time Setup

Run these commands once from the repo root:

```bash
cd /home/christopher/Documents/openpi-finetune/openpi

mkdir -p .cache/libero-openpi

cat > .cache/libero-openpi/config.yaml <<'EOF'
benchmark_root: /home/christopher/Documents/openpi-finetune/openpi/third_party/libero/libero/libero
bddl_files: /home/christopher/Documents/openpi-finetune/openpi/third_party/libero/libero/libero/bddl_files
init_states: /home/christopher/Documents/openpi-finetune/openpi/third_party/libero/libero/libero/init_files
datasets: /home/christopher/Documents/openpi-finetune/openpi/third_party/libero/libero/datasets
assets: /home/christopher/Documents/openpi-finetune/openpi/third_party/libero/libero/libero/assets
EOF
```

If the LIBERO eval environment is not already installed, set it up once:

```bash
cd /home/christopher/Documents/openpi-finetune/openpi

rm -rf examples/libero/.venv
uv venv --python 3.10 examples/libero/.venv
source examples/libero/.venv/bin/activate

uv pip install imageio tqdm tyro mujoco==3.2.3 robosuite==1.4.1 opencv-python bddl==1.0.1 \
  torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

uv pip install "numpy<2"
uv pip install future easydict cloudpickle gym==0.25.2 hydra-core==1.2.0
uv pip install matplotlib==3.5.3 wandb==0.13.1 transformers==4.21.1 robomimic==0.2.0 einops==0.4.1 thop==0.1.1-2209072238
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
```

The Python split matters here:

- `examples/libero/.venv` is the LIBERO evaluator environment and should be Python 3.10
- `scripts/serve_policy.py` must run from the root OpenPI environment with Python 3.11+ because the root project requires `>=3.11` and the script uses newer syntax

Do not run `python scripts/serve_policy.py` from `examples/libero/.venv` or you will hit `SyntaxError: invalid syntax` at `match args.policy`.
Do not use Python 3.11 for the eval venv with the pinned CUDA 11.3 torch wheels in this workflow; `torch==1.11.0+cu113` does not provide a `cp311` wheel.

## Environment Variables

Use these in the LIBERO eval window:

```bash
cd /home/christopher/Documents/openpi-finetune/openpi
source examples/libero/.venv/bin/activate

export PYTHONPATH=/home/christopher/Documents/openpi-finetune/openpi/src:/home/christopher/Documents/openpi-finetune/openpi/packages/openpi-client/src:/home/christopher/Documents/openpi-finetune/openpi/third_party/libero
export LIBERO_CONFIG_PATH=/home/christopher/Documents/openpi-finetune/openpi/.cache/libero-openpi
export USE_TF=0
```

Important:

- use `/home/christopher/Documents/openpi-finetune/openpi/src` in `PYTHONPATH` for `openpi`
- do not use the repo root alone, because `openpi` lives under `src/openpi`

For the policy-server window, do not activate `examples/libero/.venv`. Run the server from the repo root with `uv run` so it uses the OpenPI Python 3.11+ environment.

## Two-Window Workflow

### Window 1: Policy Server

```bash
cd /home/christopher/Documents/openpi-finetune/openpi
export USE_TF=0

uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi0_fast_libero_object_train7_low_mem_finetune \
  --policy.dir=checkpoints/pi0_fast_libero_object_train7_low_mem_finetune/object_train7_lora/29999
```

### Window 2: Evaluate All 10 `libero_object` Tasks

```bash
cd /home/christopher/Documents/openpi-finetune/openpi
source examples/libero/.venv/bin/activate

export PYTHONPATH=/home/christopher/Documents/openpi-finetune/openpi/src:/home/christopher/Documents/openpi-finetune/openpi/packages/openpi-client/src:/home/christopher/Documents/openpi-finetune/openpi/third_party/libero
export LIBERO_CONFIG_PATH=/home/christopher/Documents/openpi-finetune/openpi/.cache/libero-openpi
export USE_TF=0

python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.host 127.0.0.1 \
  --args.port 8000
```

That runs the full `libero_object` suite. Since `libero_object` has 10 tasks and the evaluator defaults to `50` trials per task, this is `500` rollouts total.

## Recommended Smoke Test First

Before doing the full run, start with a small test:

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.num-trials-per-task 2
```

## Held-Out 3-Task Eval for the 7/3 Split

If you want to evaluate only the held-out tasks from `data/libero/object_7train_3test.json`, use this in the eval window:

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-split-file data/libero/object_7train_3test.json \
  --args.task-split eval
```

## Running Different Checkpoints

To use a different checkpoint, update these flags in the server window:

- `--policy.config`
- `--policy.dir`

Example: full fine-tune instead of LoRA:

```bash
uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi0_fast_libero_object_train7 \
  --policy.dir=checkpoints/pi0_fast_libero_object_train7/object_train7_full/30000
```

Example: earlier LoRA checkpoint:

```bash
uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi0_fast_libero_object_train7_low_mem_finetune \
  --policy.dir=checkpoints/pi0_fast_libero_object_train7_low_mem_finetune/object_train7_lora/25000
```

Rule of thumb:

- full fine-tune checkpoints use `pi0_fast_libero_object_train7`
- LoRA checkpoints use `pi0_fast_libero_object_train7_low_mem_finetune`

## Running Different Task Suites

Change only `--args.task-suite-name` in the eval window.

Supported suites in this evaluator:

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`
- `libero_90`

Examples:

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_goal \
  --args.host 127.0.0.1 \
  --args.port 8000
```

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_10 \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.num-trials-per-task 5
```

For larger suites like `libero_10` and especially `libero_90`, reduce `--num_trials_per_task` unless you intentionally want a very large run.

## Useful Eval Arguments

The evaluator in `examples/libero/main.py` also supports task filtering.

Evaluate only specific task indices:

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-indices 0 1 2
```

Evaluate only specific task names:

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-names "pick up the milk and place it in the basket"
```

Change the rollout video output directory:

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.video-out-path data/libero/videos_object_eval
```

## Where Results Show Up

By default:

- rollout videos are written under `data/libero/videos`
- success metrics are printed to the eval terminal

Useful lines to watch in the eval output:

- `Task: ...`
- `# episodes completed so far: ...`
- `# successes: ...`
- `Current task success rate: ...`
- `Current total success rate: ...`
- `Total success rate: ...`

## Common Failures

### `ModuleNotFoundError: No module named 'libero'`

The LIBERO eval environment is not installed, or the wrong venv is active.

Fix:

- activate `examples/libero/.venv`
- install the dependencies from the setup section

### Missing or wrong `config.yaml`

If LIBERO tries to read from `~/.libero/config.yaml` or from a stale config directory, it may point at the wrong checkout.

Fix:

- ensure `LIBERO_CONFIG_PATH=/home/christopher/Documents/openpi-finetune/openpi/.cache/libero-openpi`
- verify `.cache/libero-openpi/config.yaml` exists

### `SyntaxError: invalid syntax` at `match args.policy`

This usually means the policy server was started from a stale old Python 3.8 LIBERO venv instead of the root OpenPI environment.

Fix:

- do not activate `examples/libero/.venv` in Window 1
- start the policy server with `uv run scripts/serve_policy.py ...` from the repo root
- keep `examples/libero/.venv` only for the LIBERO eval client in Window 2

### `ModuleNotFoundError` or wheel resolution failures while creating `examples/libero/.venv`

The frozen `examples/libero/requirements.txt` is stale for this workflow. On this machine, the working eval env uses Python 3.10 plus the package set from the setup block above instead of `uv pip sync`.

### Server starts but eval cannot connect

Check:

- the policy server is still running in Window 1
- both windows use port `8000`
- eval uses `--args.host 127.0.0.1`

### Wrong checkpoint config pairing

If you change `--policy.dir`, make sure `--policy.config` still matches that checkpoint family.

## Minimal Copy-Paste Version

### Window 1

```bash
cd /home/christopher/Documents/openpi-finetune/openpi
export USE_TF=0
uv run scripts/serve_policy.py --port=8000 policy:checkpoint --policy.config=pi0_fast_libero_object_train7_low_mem_finetune --policy.dir=checkpoints/pi0_fast_libero_object_train7_low_mem_finetune/object_train7_lora/29999
```

### Window 2

```bash
cd /home/christopher/Documents/openpi-finetune/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=/home/christopher/Documents/openpi-finetune/openpi/src:/home/christopher/Documents/openpi-finetune/openpi/packages/openpi-client/src:/home/christopher/Documents/openpi-finetune/openpi/third_party/libero
export LIBERO_CONFIG_PATH=/home/christopher/Documents/openpi-finetune/openpi/.cache/libero-openpi
export USE_TF=0
python examples/libero/main.py --args.task-suite-name libero_object --args.host 127.0.0.1 --args.port 8000 --args.num-trials-per-task 2
```
