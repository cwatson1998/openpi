# LIBERO Eval via tmux

This repo now includes a small tmux-based workflow for serving a checkpoint and running LIBERO evaluation in one command.

The main entrypoints are:

- [`scripts/eval_libero_tmux.sh`](../scripts/eval_libero_tmux.sh)
- [`scripts/watch_libero_tmux.sh`](../scripts/watch_libero_tmux.sh)

The launcher creates a tmux session with two windows:

- `server`: serves the checkpoint with [`scripts/serve_policy.py`](../scripts/serve_policy.py)
- `eval`: runs the LIBERO evaluator from [`examples/libero/main.py`](../examples/libero/main.py)

## Defaults

If you run the launcher with no arguments, it is configured for the finished LoRA LIBERO-Object run in this repo:

- policy config: `pi0_fast_libero_object_train7_low_mem_finetune`
- checkpoint: `checkpoints/pi0_fast_libero_object_train7_low_mem_finetune/object_train7_lora/29999`
- task suite: `libero_object`
- trials per task: `50`
- tmux session name: `libero-eval`

## Quick Start

From the repo root:

```bash
./scripts/eval_libero_tmux.sh
```

That will:

1. use the main OpenPI env from `.venv` for the policy server
2. create `examples/libero/.venv` if needed
3. install the LIBERO evaluation dependencies if needed
4. write a repo-local LIBERO config override under `.cache/libero-openpi`
5. create a tmux session
6. start the policy server
7. start the evaluator against that server

Attach manually later with:

```bash
tmux attach -t libero-eval
```

Create the session without attaching:

```bash
./scripts/eval_libero_tmux.sh --detach
```

## Monitoring

To watch the run in a compact summary view:

```bash
./scripts/watch_libero_tmux.sh
```

One-shot snapshot:

```bash
./scripts/watch_libero_tmux.sh --once
```

Watch a different session:

```bash
./scripts/watch_libero_tmux.sh --session object10
```

This watcher reads the `server` and `eval` tmux panes and summarizes:

- the latest task description
- completed episode count
- cumulative successes
- current task success rate
- current total success rate
- recent server output

## Switching Checkpoints

To evaluate a different checkpoint, override both the policy config and the checkpoint directory.

Example: evaluate the full fine-tune instead of the LoRA run:

```bash
./scripts/eval_libero_tmux.sh \
  --policy-config pi0_fast_libero_object_train7 \
  --checkpoint-dir checkpoints/pi0_fast_libero_object_train7/object_train7_full/30000
```

Example: evaluate an earlier LoRA checkpoint:

```bash
./scripts/eval_libero_tmux.sh \
  --checkpoint-dir checkpoints/pi0_fast_libero_object_train7_low_mem_finetune/object_train7_lora/25000
```

The important rule is:

- `--policy-config` must match the training config that produced the checkpoint

For the prepared object-train7 configs in this repo:

- `pi0_fast_libero_object_train7` matches full fine-tuning checkpoints
- `pi0_fast_libero_object_train7_low_mem_finetune` matches LoRA / low-memory checkpoints

## Switching Task Suites

The LIBERO evaluator supports:

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`
- `libero_90`

Run a different suite with `--suite`.

Example:

```bash
./scripts/eval_libero_tmux.sh --suite libero_goal
```

Example:

```bash
./scripts/eval_libero_tmux.sh --suite libero_10
```

The evaluator itself sets different rollout horizons per suite in [`examples/libero/main.py`](../examples/libero/main.py), so no extra max-step flag is needed.

## Changing Trial Count

The default is `50` rollouts per task, which can be expensive for large suites.

Short smoke test:

```bash
./scripts/eval_libero_tmux.sh --trials 2
```

Medium run:

```bash
./scripts/eval_libero_tmux.sh --trials 10
```

## Passing Extra Evaluator Args

Any arguments after `--` are passed directly to [`examples/libero/main.py`](../examples/libero/main.py). Because that script uses `tyro.cli(eval_libero)`, the flags need the `--args.` prefix.

Evaluate only specific task indices:

```bash
./scripts/eval_libero_tmux.sh -- --args.task-indices 0 1 2
```

Change the output directory for rollout videos:

```bash
./scripts/eval_libero_tmux.sh -- --args.video-out-path data/libero/videos_goal
```

Change the seed:

```bash
./scripts/eval_libero_tmux.sh -- --args.seed 11
```

Use a split file and evaluate only its `eval` subset:

```bash
./scripts/eval_libero_tmux.sh -- \
  --args.task-split-file data/libero/object_7train_3test.json \
  --args.task-split eval
```

## Running Multiple Sessions

Use a distinct session name and port for each concurrent run.

Example:

```bash
./scripts/eval_libero_tmux.sh \
  --session goal-eval \
  --port 8010 \
  --suite libero_goal \
  --trials 10
```

Then monitor it with:

```bash
./scripts/watch_libero_tmux.sh --session goal-eval
```

## Common Operations

Attach to the session:

```bash
tmux attach -t libero-eval
```

List sessions:

```bash
tmux ls
```

Kill the session:

```bash
tmux kill-session -t libero-eval
```

Capture the recent eval pane output:

```bash
tmux capture-pane -pt libero-eval:eval -S -200
```

Capture the recent server pane output:

```bash
tmux capture-pane -pt libero-eval:server -S -80
```

## Environment Notes

The launcher deliberately overrides `LIBERO_CONFIG_PATH` to avoid accidentally using a different local LIBERO checkout from `~/.libero/config.yaml`.

It writes:

- config dir: `.cache/libero-openpi`
- config file: `.cache/libero-openpi/config.yaml`

The venv used by default is:

- `examples/libero/.venv`

It should be created with Python 3.10 for this workflow.

After installing the LIBERO requirements, make sure the env still has
`pkg_resources` available:

```bash
examples/libero/.venv/bin/python -m ensurepip --upgrade
examples/libero/.venv/bin/python -m pip install --upgrade 'setuptools<81'
examples/libero/.venv/bin/python - <<'PY'
import pkg_resources
print("pkg_resources ok")
PY
```

This is currently necessary because W&B still imports `pkg_resources`, while
newer setuptools releases remove it.

The launcher exports:

- `PYTHONPATH=$REPO/src:$REPO/packages/openpi-client/src:$REPO/third_party/libero`

That `src` entry is important because `openpi` is a `src/` layout package in this repo.

The policy server runs in the repo's main OpenPI environment by default:

- `.venv`

If you touch [`examples/libero/main.py`](../examples/libero/main.py),
keep it Python-3.10-compatible. The local/tmux LIBERO eval flow still runs
that file from the separate Python 3.10 simulator environment, not from the
main Python 3.11 repo environment.

This split is intentional: the LIBERO eval workflow currently needs a Python 3.10 environment because the pinned CUDA 11.3 torch wheels do not support Python 3.11, while the OpenPI project itself requires Python 3.11 for the policy server.

If you already prepared that environment yourself and do not want the launcher to touch it, use:

```bash
./scripts/eval_libero_tmux.sh --setup-env never
```

If you want to force a fresh dependency setup step in that venv, use:

```bash
./scripts/eval_libero_tmux.sh --setup-env always
```

## Why Not `~/.libero`?

LIBERO reads `config.yaml` from the directory pointed to by `LIBERO_CONFIG_PATH`, or from `~/.libero` if that env var is unset.

On this machine, `~/.libero/config.yaml` was already pointing at a different LIBERO checkout. Using it would make this repo's evaluator load assets, BDDL files, and init states from the wrong tree.

So yes, overriding the config path is the right thing to do. What was wrong in the first version was the choice of a `/tmp` default plus a directory-creation bug. The launcher now uses a repo-local cache directory instead:

- stable across sessions
- easy to inspect
- tied to this checkout instead of global machine state
