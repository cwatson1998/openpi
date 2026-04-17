# Partial Eval Report: `6s4mowfn`

This note records what happened to the crashed LIBERO eval run:

- W&B run: `penn-pal/libero/6s4mowfn`
- Eval name: `libero10_logic_full_from_libero_ckpt_20260414_174205_29999_libero_10_wsfix`
- Slurm job: `445151`
- Host: `dj-l40-0.grasp.maas`
- Checkpoint: `checkpoints/pi0_fast_libero_10_logic/libero10_logic_full_from_libero_ckpt_20260414_174205/29999`

## Outcome

- This run did not fail at startup.
- It ran for essentially the full six-hour Slurm wall-clock limit.
- Slurm killed it due to `TIME LIMIT`.
- Because the code revision used by the run only wrote `results.json` at the very end, there is no final `results.json` on disk for this run.

## Timing

- Slurm start: `2026-04-16 00:18:27`
- Slurm end: `2026-04-16 06:18:35`
- Slurm state: `TIMEOUT`
- W&B init time from metadata: `2026-04-16T04:19:15Z`
- W&B runtime near `5h59m` is expected because W&B starts after some job setup, while Slurm counts the full batch job.

## Partial Progress Recovered

The eval completed `70` episodes with `51` successes before timing out.

- Running total success rate at crash: `72.9%`
- Fully completed tasks: `7 / 10`
- It was killed at the start of task 8, episode 1

Recovered completed-task results from the log:

1. `(And (In alphabet_soup basket) (In tomato_sauce basket))`: `10 / 10`
2. `(And (In cream_cheese basket) (In butter basket))`: `10 / 10`
3. `(And (Turnon flat_stove) (On moka_pot flat_stove))`: `1 / 10`
4. `(And (Close white_cabinet) (In akita_black_bowl white_cabinet))`: `5 / 10`
5. `(And (On porcelain_mug plate) (On white_yellow_mug plate))`: `10 / 10`
6. `(And (In black_book desk_caddy))`: `9 / 10`
7. `(And (On porcelain_mug plate) (On chocolate_pudding living_room_table_plate_right))`: `6 / 10`

## Files Left Behind

The run output directory contains only rollout videos:

- [data/libero/evals/libero10_logic_full_from_libero_ckpt_20260414_174205_29999_libero_10_wsfix](/home/chriswatson/openpi-finetune/openpi/data/libero/evals/libero10_logic_full_from_libero_ckpt_20260414_174205_29999_libero_10_wsfix:1)

There is no final:

- `results.json`
- `eval_progress.json`

That absence is consistent with the code revision used by this run.

## Why No JSON Was Saved

W&B metadata shows this run used git commit `933119bf8d38664dfb861b1e3ea612e2d8054cd6`.

In that revision of [examples/libero/main.py](/home/chriswatson/openpi-finetune/openpi/examples/libero/main.py:1), the eval wrote `results.json` only once, at the very end of the full run. Since the job timed out before all `10 x 10 = 100` episodes completed, it never reached that final write.

## Evidence

- Slurm log: [logs/slurm/openpi_libero_eval-445151.out](/home/chriswatson/openpi-finetune/openpi/logs/slurm/openpi_libero_eval-445151.out:1)
- W&B metadata: [wandb/run-20260416_041915-6s4mowfn/files/wandb-metadata.json](/home/chriswatson/openpi-finetune/openpi/wandb/run-20260416_041915-6s4mowfn/files/wandb-metadata.json:1)
- Recovery workflow: [CRASHED_EVAL_RECOVERY_GUIDE.md](/home/chriswatson/openpi-finetune/openpi/CRASHED_EVAL_RECOVERY_GUIDE.md:1)
