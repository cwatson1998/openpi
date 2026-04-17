# Crashed Eval Recovery Guide

This guide shows how to recover partial progress from a crashed or timed-out LIBERO eval, even when the run did not write a final `results.json`.

It applies to evals launched through:

- [examples/libero/eval_checkpoint.slurm](/home/chriswatson/openpi-finetune/openpi/examples/libero/eval_checkpoint.slurm:1)
- [scripts/submit_libero_eval_slurm.sh](/home/chriswatson/openpi-finetune/openpi/scripts/submit_libero_eval_slurm.sh:1)
- [examples/libero/main.py](/home/chriswatson/openpi-finetune/openpi/examples/libero/main.py:1)

## What To Recover

For a crashed eval, usually you want to answer five questions:

1. Which Slurm job and W&B run correspond to the eval?
2. Did it fail immediately, or did it run for hours?
3. How many episodes completed before the crash?
4. What was the running success rate at the point of failure?
5. Which task was in progress when it died?

## Step 1: Identify The Exact Run

Start from any one of:

- W&B run id like `6s4mowfn`
- eval name like `libero10_logic_full_from_libero_ckpt_..._wsfix`
- Slurm job id like `445151`

Useful repo-local search:

```bash
rg -n "6s4mowfn|libero10_logic_full_from_libero_ckpt_20260414_174205_29999_libero_10_wsfix|445151" \
  -S logs/slurm EXPERIMENTS_APRIL_14.md wandb
```

This usually tells you:

- the matching Slurm log file
- the output directory
- the checkpoint used
- the W&B local run directory

## Step 2: Check Slurm Accounting First

Use `sacct` to determine whether the job failed early, timed out, was cancelled, or completed:

```bash
sacct -j 445151 \
  --format=JobID,JobName%35,Partition,State,ExitCode,Elapsed,Start,End,NodeList,AllocTRES%40,Reason%30
```

Interpretation:

- `TIMEOUT`: the job used its full wall-clock allocation and Slurm killed it
- `FAILED`: the job exited with an application error
- `CANCELLED`: user or cluster intervention stopped it
- `COMPLETED`: the batch job finished normally

Important detail:

- Slurm elapsed time includes full batch-job setup
- W&B runtime starts later, around `wandb.init()`
- a small mismatch between Slurm runtime and W&B runtime is normal

## Step 3: Read The Slurm Log

The Slurm log is the primary source for partial progress.

Open the full log:

```bash
sed -n '1,260p' logs/slurm/openpi_libero_eval-445151.out
sed -n '260,999p' logs/slurm/openpi_libero_eval-445151.out
```

Or inspect the tail:

```bash
tail -n 120 logs/slurm/openpi_libero_eval-445151.out
```

What to look for:

- server startup succeeded
- checkpoint restore succeeded
- websocket connection opened
- repeated episode logs like:
  - `Task: ...`
  - `Starting episode N...`
  - `Success: True/False`
  - `# episodes completed so far: ...`
  - `# successes: ...`
- per-task rollups like:
  - `Current task success rate: ...`
  - `Current total success rate: ...`
- final Slurm termination line like:
  - `CANCELLED ... DUE TO TIME LIMIT`

## Step 4: Recover Progress From The Log

If there is no final JSON, reconstruct progress from the latest log lines.

The most important lines are:

- `# episodes completed so far: X`
- `# successes: Y (...)`
- `Current task success rate: ...`
- `Current total success rate: ...`

These tell you:

- how many episodes completed
- how many succeeded
- the running total success rate
- which tasks fully completed

To scan just those lines:

```bash
rg -n "# episodes completed so far|# successes:|Current task success rate|Current total success rate|Task: " \
  logs/slurm/openpi_libero_eval-445151.out
```

To find the last completed point before the crash:

```bash
tail -n 200 logs/slurm/openpi_libero_eval-445151.out
```

## Step 5: Check The Output Directory

Look at the eval output directory:

```bash
find data/libero/evals/<eval_name> -maxdepth 1 -printf '%f\n' | sort
```

Possible outcomes:

- `results.json` exists:
  the run reached the final write path
- `eval_progress.json` exists:
  the run used newer incremental persistence and you can read partial results directly
- only rollout videos exist:
  recover progress from the log instead

Example:

```bash
find data/libero/evals/libero10_logic_full_from_libero_ckpt_20260414_174205_29999_libero_10_wsfix \
  -maxdepth 1 -printf '%f\n' | sort
```

## Step 6: Check W&B Metadata

Local W&B metadata is useful for confirming:

- host node
- Slurm job id
- command-line args
- checkpoint dir
- git commit actually used by the run

Open it with:

```bash
sed -n '1,220p' wandb/run-20260416_041915-6s4mowfn/files/wandb-metadata.json
```

This can resolve confusion when:

- W&B and Slurm seem to disagree on runtime
- there are multiple similar eval names
- the working tree has changed since the run

## Step 7: Check Which Code Revision Ran

This matters because older eval code may only write final outputs at the end.

The W&B metadata includes the git commit used by the run. Compare that commit with your current working tree:

```bash
git show <commit>:examples/libero/main.py | rg -n "results_out_path|eval_progress|write_text"
git diff -- examples/libero/main.py
```

Interpretation:

- if the old revision only writes `results.json` once at the end, a timed-out run may leave no JSON at all
- if the newer revision writes `eval_progress.json` incrementally, partial recovery should be easier

## Fast Recovery Checklist

When an eval crashes, run these in order:

```bash
rg -n "<wandb_id>|<eval_name>|<job_id>" -S logs/slurm wandb EXPERIMENTS_APRIL_14.md
sacct -j <job_id> --format=JobID,JobName%35,State,ExitCode,Elapsed,Start,End,NodeList,Reason%30
tail -n 120 logs/slurm/<logfile>.out
find data/libero/evals/<eval_name> -maxdepth 1 -printf '%f\n' | sort
sed -n '1,220p' wandb/run-<timestamp>-<wandb_id>/files/wandb-metadata.json
```

## Example: `6s4mowfn`

For `penn-pal/libero/6s4mowfn`, the recovered story was:

- Slurm job `445151`
- host `dj-l40-0.grasp.maas`
- state `TIMEOUT`
- elapsed about six hours
- `70` episodes completed
- `51` successes
- running success rate `72.9%`
- finished `7 / 10` tasks
- died at the start of task 8
- no final `results.json` because that run used an older code revision that only wrote final output at the end

See the saved incident report:

- [EVAL_6s4mowfn_PARTIAL_REPORT.md](/home/chriswatson/openpi-finetune/openpi/EVAL_6s4mowfn_PARTIAL_REPORT.md:1)

## Practical Recommendation

For future long evals:

- prefer incremental output persistence during the run
- keep the Slurm log
- keep the local W&B run directory
- if an eval may exceed six hours, increase the Slurm `--time` limit before launching

