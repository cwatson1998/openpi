# LIBERO Eval via Slurm

This repo includes a Slurm-friendly LIBERO checkpoint eval flow:

- [examples/libero/eval_checkpoint.slurm](../examples/libero/eval_checkpoint.slurm)
- [scripts/submit_libero_eval_slurm.sh](../scripts/submit_libero_eval_slurm.sh)

The batch job runs both pieces inside one allocation:

1. starts the OpenPI policy server with [scripts/serve_policy.py](../scripts/serve_policy.py)
2. waits for the server to accept connections
3. runs the LIBERO evaluator with [examples/libero/main.py](../examples/libero/main.py)

The eval job logs to W&B by default under project `libero`.
The default metadata is derived from the source checkpoint:

- run name: `<eval-name>`
- group: `<train-run-name>`
- tags include:
  `eval`, `libero`, suite, prompt style, policy config, source train run, source checkpoint step

During the run, the evaluator now also:

- streams per-episode running metrics to W&B as episodes complete
- writes incremental state to both `results.json` and `eval_progress.json`
- uploads a bounded set of example rollout videos to W&B while the job is still running
  currently the first success, first failure, and first error video per task

The submit helper defaults to a conservative 24G+ VRAM node list:

- `dj-a40-0.grasp.maas`
- `dj-a40-1.grasp.maas`
- `dj-l40-0.grasp.maas`

It also auto-derives a per-eval server port when you do not pass `--port`, so multiple eval jobs can run on the same node without colliding on `8020`.

## Basic Usage

Logic-prompt checkpoint:

```bash
./scripts/submit_libero_eval_slurm.sh \
  --policy-config pi0_fast_libero_10_logic \
  --checkpoint-dir checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_20260414_174057/23600
```

Default-English checkpoint:

```bash
./scripts/submit_libero_eval_slurm.sh \
  --policy-config pi0_fast_libero \
  --checkpoint-dir checkpoints/pi0_fast_libero/libero10_default_ctrl_20260414_174400/200 \
  --suite libero_10
```

## Important Rule

`--policy-config` must match the training config that produced the checkpoint.

- full logic runs in this repo use `pi0_fast_libero_10_logic`
- default-English control runs use `pi0_fast_libero`

## Environment Setup

By default the submit helper uses `--setup-env auto`.

That means the batch job will repair or create `examples/libero/.venv` when needed.
If you want to skip all bootstrap work and require an already-good env, use `--setup-env never`.

The `never` mode assumes:

- OpenPI server env exists at `.venv`
- LIBERO eval env exists at `examples/libero/.venv`

If you want to force a rebuild/repair of the LIBERO eval env, use:

```bash
./scripts/submit_libero_eval_slurm.sh ... --setup-env always
```

## Monitoring

Queue status:

```bash
/bin/bash -lc "squeue -u \"$USER\" -o \"%8i %9P %30j %2t %10M %6D %20R\""
```

Batch accounting:

```bash
sacct -j <jobid> --parsable2 --noheader --format=JobID,JobName%40,Partition,State,ExitCode,Elapsed,NodeList%30,Submit,Start,End
```

Log output:

```bash
tail -n 100 logs/slurm/openpi_libero_eval-<jobid>.out
```

W&B:

- eval runs appear in project `libero`
- the run name should match the eval name
- the run group should match the source training run name
- tags show suite, prompt type, policy config, and checkpoint step
- running charts should update per episode rather than only per completed task
- selected rollout videos may appear before the full eval finishes

Eval artifacts are written under:

```bash
data/libero/evals/<eval-name>/
```

Key files inside that directory:

```bash
data/libero/evals/<eval-name>/results.json
data/libero/evals/<eval-name>/eval_progress.json
```
