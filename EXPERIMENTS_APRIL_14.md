# Experiments April 14

Updated on 2026-04-16 after the April 14 training launches completed and the follow-up eval work was repaired, resubmitted, and brought past the earlier websocket timeout failure mode.

## Summary

The main April 14 logic-prompt training runs both completed successfully on April 15:

1. `444315`:
   full fine-tune on `libero_10` using the logic task descriptions, starting from the base `pi0-fast` checkpoint, batch size `8`.
2. `444316`:
   full fine-tune on `libero_10` using the logic task descriptions, starting from the released fine-tuned LIBERO checkpoint, batch size `8`.

The default-English control run did not complete:

3. `444318`:
   control experiment using the normal English task descriptions on `libero_10`, starting from the base `pi0-fast` checkpoint, batch size `8`, `4` GPUs.
   This run failed earlier and did not produce the comparison result we wanted.

The norm-stats prerequisite for the logic runs was generated successfully earlier on April 14:

- `444305`: `openpi_libero10_logic_normstats`
- status: `COMPLETED`
- output file: [assets/pi0_fast_libero_10_logic/physical-intelligence/libero/norm_stats.json](/home/chriswatson/openpi-finetune/openpi/assets/pi0_fast_libero_10_logic/physical-intelligence/libero/norm_stats.json:1)

There is also a new Slurm-based LIBERO eval path now:

- batch script: [examples/libero/eval_checkpoint.slurm](/home/chriswatson/openpi-finetune/openpi/examples/libero/eval_checkpoint.slurm:1)
- submit helper: [scripts/submit_libero_eval_slurm.sh](/home/chriswatson/openpi-finetune/openpi/scripts/submit_libero_eval_slurm.sh:1)
- docs: [docs/libero_eval_slurm.md](/home/chriswatson/openpi-finetune/openpi/docs/libero_eval_slurm.md:1)

## Final Training Runs

### 444315

- Job name: `openpi_libero10_logic_full`
- Config: `pi0_fast_libero_10_logic`
- Prompt type: logic task descriptions from [data/libero/libero_10_logic_descriptions.json](/home/chriswatson/openpi-finetune/openpi/data/libero/libero_10_logic_descriptions.json:1)
- Weight init: base `pi0-fast`
- Batch size: `8`
- FSDP devices: `2`
- Final node: `dj-a40-1.grasp.maas`
- State: `COMPLETED`
- Runtime: `01:26:26`
- Checkpoint dir: [checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_20260414_174057](/home/chriswatson/openpi-finetune/openpi/checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_20260414_174057:1)
- Final checkpoint: [checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_20260414_174057/29999](/home/chriswatson/openpi-finetune/openpi/checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_20260414_174057/29999:1)
- W&B: `https://wandb.ai/penn-pal/libero/runs/psiaujqp`
- Log: [logs/slurm/openpi_libero10_logic_full-444315.out](/home/chriswatson/openpi-finetune/openpi/logs/slurm/openpi_libero10_logic_full-444315.out:1)

Submit command used:

```bash
sbatch --partition=dineshj-compute --qos=dj-med --export=ALL,EXP_NAME=libero10_logic_full_20260414_174057,WANDB_PROJECT=libero,BATCH_SIZE=8 examples/libero/train_libero_10_logic_full.slurm
```

Training command inside the job:

```bash
uv run --frozen --no-dev scripts/train.py pi0_fast_libero_10_logic --exp-name=libero10_logic_full_20260414_174057 --project-name=libero --fsdp-devices=2 --batch-size=8 --resume --save-interval=200 --keep-period=1000 --num-workers=8
```

### 444316

- Job name: `openpi_libero10_logic_full`
- Config: `pi0_fast_libero_10_logic`
- Prompt type: logic task descriptions from [data/libero/libero_10_logic_descriptions.json](/home/chriswatson/openpi-finetune/openpi/data/libero/libero_10_logic_descriptions.json:1)
- Weight init: released fine-tuned LIBERO checkpoint via `s3://openpi-assets/checkpoints/pi0_fast_libero/params`
- Batch size: `8`
- FSDP devices: `2`
- Final node: `dj-l40-0.grasp.maas`
- State: `COMPLETED`
- Runtime: `02:09:57`
- Checkpoint dir: [checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_from_libero_ckpt_20260414_174205](/home/chriswatson/openpi-finetune/openpi/checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_from_libero_ckpt_20260414_174205:1)
- Final checkpoint: [checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_from_libero_ckpt_20260414_174205/29999](/home/chriswatson/openpi-finetune/openpi/checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_from_libero_ckpt_20260414_174205/29999:1)
- W&B: `https://wandb.ai/penn-pal/libero/runs/p4gyv61w`
- Log: [logs/slurm/openpi_libero10_logic_full-444316.out](/home/chriswatson/openpi-finetune/openpi/logs/slurm/openpi_libero10_logic_full-444316.out:1)

Submit command used:

```bash
sbatch --partition=dineshj-compute --qos=dj-med --nodelist=dj-l40-0.grasp.maas --export=ALL,EXP_NAME=libero10_logic_full_from_libero_ckpt_20260414_174205,WANDB_PROJECT=libero,BATCH_SIZE=8,WEIGHT_LOADER_PARAMS_PATH=s3://openpi-assets/checkpoints/pi0_fast_libero/params examples/libero/train_libero_10_logic_full.slurm
```

Training command inside the job:

```bash
uv run --frozen --no-dev scripts/train.py pi0_fast_libero_10_logic --exp-name=libero10_logic_full_from_libero_ckpt_20260414_174205 --project-name=libero --fsdp-devices=2 --batch-size=8 --resume --save-interval=200 --keep-period=1000 --num-workers=8 --weight-loader.params-path=s3://openpi-assets/checkpoints/pi0_fast_libero/params
```

### 444318

- Job name: `openpi_libero10_default_ctrl`
- Config: `pi0_fast_libero`
- Prompt type: default English dataset task descriptions
- Task subset: `libero_10`
- Weight init: base `pi0-fast`
- Batch size: `8`
- FSDP devices: `4`
- GPUs requested: `4`
- CPUs requested: `8`
- Requested node: `dj-a40-0.grasp.maas`
- State: failed earlier
- Checkpoint / experiment name: `libero10_default_ctrl_20260414_174400`
- Log: [logs/slurm/openpi_libero10_default_ctrl-444318.out](/home/chriswatson/openpi-finetune/openpi/logs/slurm/openpi_libero10_default_ctrl-444318.out:1)

Submit command used:

```bash
sbatch --partition=dineshj-compute --qos=dj-med --job-name=openpi_libero10_default_ctrl --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:4 --nodelist=dj-a40-0.grasp.maas --time=12:00:00 --output=logs/slurm/%x-%j.out --wrap 'cd /home/chriswatson/openpi-finetune/openpi && export USE_TF=0 && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 && export TOKENIZERS_PARALLELISM=false && export WANDB_MODE=online && uv run --frozen --no-dev scripts/train.py pi0_fast_libero --exp-name=libero10_default_ctrl_20260414_174400 --project-name=libero --fsdp-devices=4 --batch-size=8 --resume --save-interval=200 --keep-period=1000 --num-workers=4 --data.task-suite-name=libero_10 --data.assets.assets-dir=./assets/pi0_fast_libero_10_logic'
```

Notes:

- This run uses normal English task descriptions because it uses `pi0_fast_libero` and does not set `task_description_path`.
- The assets override points at `./assets/pi0_fast_libero_10_logic` only to reuse the already-computed normalization stats.

## Eval Runs

The LIBERO eval path required several rounds of repair before becoming usable:

- added repo-local Slurm eval tooling
- added W&B logging for eval runs
- fixed missing `third_party/libero` submodule checkout
- fixed port collisions between concurrent eval jobs by auto-deriving ports
- tightened the env readiness check so incomplete eval envs are not treated as valid
- switched the bootstrap back to a fuller LIBERO dependency spec instead of the earlier over-trimmed package list

Earlier eval attempts failed for real setup reasons:

- missing `examples/libero/.venv`
- missing `third_party/libero` checkout
- missing `matplotlib`
- port collision when two eval jobs landed on the same node

The first round of eval launches (`445143`, `445144`) established that the Slurm tooling and dependency bootstrap were mostly working, but they still ran into runtime websocket timeout behavior. A later pair (`445147`, `445148`) reproduced the same problem after more of the environment issues had already been fixed.

The durable fix was in the websocket transport itself:

- [src/openpi/serving/websocket_policy_server.py](/home/chriswatson/openpi-finetune/openpi/src/openpi/serving/websocket_policy_server.py:1) now runs `policy.infer()` in a worker thread via `asyncio.to_thread(...)` so long JAX/XLA inference does not block the asyncio loop from answering websocket keepalive traffic.
- [packages/openpi-client/src/openpi_client/websocket_client_policy.py](/home/chriswatson/openpi-finetune/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py:1) now uses more forgiving websocket open/close/ping timeouts.

Current replacement eval jobs:

### 445150

- Source training run: `444315`
- Checkpoint: [checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_20260414_174057/29999](/home/chriswatson/openpi-finetune/openpi/checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_20260414_174057/29999:1)
- Eval name: `libero10_logic_full_20260414_174057_29999_libero_10_wsfix`
- State: `RUNNING`
- Node: `dj-a40-0.grasp.maas`
- W&B project: `libero`
- W&B group: `libero10_logic_full_20260414_174057`
- W&B: `https://wandb.ai/penn-pal/libero/runs/1ggpafl9`
- Results dir: [data/libero/evals/libero10_logic_full_20260414_174057_29999_libero_10_wsfix](/home/chriswatson/openpi-finetune/openpi/data/libero/evals/libero10_logic_full_20260414_174057_29999_libero_10_wsfix:1)
- Log: [logs/slurm/openpi_libero_eval-445150.out](/home/chriswatson/openpi-finetune/openpi/logs/slurm/openpi_libero_eval-445150.out:1)

Submit command used:

```bash
./scripts/submit_libero_eval_slurm.sh --checkpoint-dir checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_20260414_174057/29999 --name libero10_logic_full_20260414_174057_29999_libero_10_wsfix --setup-env always --wandb-project libero
```

### 445151

- Source training run: `444316`
- Checkpoint: [checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_from_libero_ckpt_20260414_174205/29999](/home/chriswatson/openpi-finetune/openpi/checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_from_libero_ckpt_20260414_174205/29999:1)
- Eval name: `libero10_logic_full_from_libero_ckpt_20260414_174205_29999_libero_10_wsfix`
- State: `RUNNING`
- Node: `dj-l40-0.grasp.maas`
- W&B project: `libero`
- W&B group: `libero10_logic_full_from_libero_ckpt_20260414_174205`
- W&B: `https://wandb.ai/penn-pal/libero/runs/6s4mowfn`
- Results dir: [data/libero/evals/libero10_logic_full_from_libero_ckpt_20260414_174205_29999_libero_10_wsfix](/home/chriswatson/openpi-finetune/openpi/data/libero/evals/libero10_logic_full_from_libero_ckpt_20260414_174205_29999_libero_10_wsfix:1)
- Log: [logs/slurm/openpi_libero_eval-445151.out](/home/chriswatson/openpi-finetune/openpi/logs/slurm/openpi_libero_eval-445151.out:1)

Submit command used:

```bash
./scripts/submit_libero_eval_slurm.sh --checkpoint-dir checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_from_libero_ckpt_20260414_174205/29999 --name libero10_logic_full_from_libero_ckpt_20260414_174205_29999_libero_10_wsfix --setup-env always --wandb-project libero
```

### Current Eval Status

As of the latest check on April 16:

- `445150` and `445151` are both still `RUNNING`
- both are running on the intended 24G+ VRAM nodes
- both rebuilt clean job-local Python 3.10 envs under `/tmp`
- both restored checkpoints, started the policy server, and established the websocket connection
- both got past the earlier failure point and reached `Starting episode 1...`

Important current interpretation:

- the old `keepalive ping timeout` / `Caught exception` signature is absent from the replacement job logs
- that makes these runs materially healthier than the superseded eval attempts
- they still need to run longer to prove full rollout completion, but they no longer appear stuck on the earlier transport failure mode

## How To Check On Them

Queue status:

```bash
/bin/bash -lc "squeue -u \"$USER\" -o \"%8i %9P %30j %2t %10M %6D %20R\""
```

Per-job accounting:

```bash
sacct -j 444315,444316,444318,445147,445148,445150,445151 --parsable2 --noheader --format=JobID,JobName%50,Partition,State,ExitCode,Elapsed,NodeList%30,Submit,Start,End
```

Tail the logs:

```bash
tail -n 80 logs/slurm/openpi_libero10_logic_full-444315.out
tail -n 80 logs/slurm/openpi_libero10_logic_full-444316.out
tail -n 80 logs/slurm/openpi_libero10_default_ctrl-444318.out
tail -n 80 logs/slurm/openpi_libero_eval-445150.out
tail -n 80 logs/slurm/openpi_libero_eval-445151.out
```

Check whether a training checkpoint was written:

```bash
find checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_20260414_174057 -maxdepth 2 -type d | sort
find checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_from_libero_ckpt_20260414_174205 -maxdepth 2 -type d | sort
find checkpoints/pi0_fast_libero/libero10_default_ctrl_20260414_174400 -maxdepth 2 -type d | sort
```

Check whether eval outputs landed:

```bash
find data/libero/evals/libero10_logic_full_20260414_174057_29999_libero_10_wsfix -maxdepth 2 | sort
find data/libero/evals/libero10_logic_full_from_libero_ckpt_20260414_174205_29999_libero_10_wsfix -maxdepth 2 | sort
```

## Commentary

- The key April 14 result is that the two logic-prompt full-finetune runs now both finished successfully with batch size `8`. The earlier blockers were not intrinsic to the task descriptions; they were a sequence of launcher/config issues: `uv` resolution, bad W&B CLI flag syntax, prompt validation against the wrong task catalog, missing norm stats, and then OOM at the original batch size `32`.
- The data-loader fix that made the logic prompt file usable is in [src/openpi/training/data_loader.py](/home/chriswatson/openpi-finetune/openpi/src/openpi/training/data_loader.py:1), and the regression test is in [src/openpi/training/data_loader_test.py](/home/chriswatson/openpi-finetune/openpi/src/openpi/training/data_loader_test.py:1).
- The task-description file itself was never the real problem. It lives at [data/libero/libero_10_logic_descriptions.json](/home/chriswatson/openpi-finetune/openpi/data/libero/libero_10_logic_descriptions.json:1) and was already correct for the `libero_10` subset.
- The eval path needed much more cleanup than the training path. The repo now has better eval tooling than it did on April 14, but this part of the stack was clearly under-specified before the debugging work in this session.
- The biggest thing we learned is that “just a `uv` env” was the right framing, but only if the environment is actually specified declaratively and built in isolation. The unstable pattern was a shared mutable `examples/libero/.venv` plus a hand-maintained install sequence. The stable pattern was:
  - explicit LIBERO eval requirements in-repo
  - a fresh per-job Python 3.10 env under `/tmp`
  - editable installs for `openpi-client` and `third_party/libero`
  - a modern `wandb` version instead of the old `0.13.1` runtime
- Another thing we learned is that the remaining eval risk moved from Python packaging into runtime serving behavior. The important fix here was not another dependency tweak; it was making the websocket server tolerant of slow JAX/XLA inference by moving `policy.infer()` off the asyncio event loop and relaxing transport timeouts on both sides.
- The top-level README command for `compute_norm_stats.py` is stale for this repo state; the script currently expects `--config-name`.

## Likely Next Steps

1. Let `445150` and `445151` run long enough to confirm real rollout completion.
   The immediate question is no longer “can they start?” but “do they continue through episodes and write stable outputs without regressing into a slower transport or inference failure later in the run?”

2. Reduce first-step latency on the server side.
   The logs show expensive XLA/cuDNN algorithm search on the first real inference step. Likely options are:
   - warm up the model with a dummy inference before starting rollouts
   - tune or disable the most expensive autotuning path if there is a safe config knob for it
   - keep the server alive across more episodes/checkpoints once warm

3. If runtime instability returns, patch client and server together again rather than only one side.
   The right place to treat this remains the serving boundary:
   - [examples/libero/main.py](/home/chriswatson/openpi-finetune/openpi/examples/libero/main.py:1)
   - [scripts/serve_policy.py](/home/chriswatson/openpi-finetune/openpi/scripts/serve_policy.py:1)
   - [packages/openpi-client](/home/chriswatson/openpi-finetune/openpi/packages/openpi-client:1)

4. Once logic eval is stable, rerun the missing default-English control.
   The clean comparison we still want is:
   - logic prompts from base `pi0-fast`
   - logic prompts from released LIBERO checkpoint
   - default English prompts from base `pi0-fast`
   Right now the control side is still incomplete because `444318` failed earlier.
