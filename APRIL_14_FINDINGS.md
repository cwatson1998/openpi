# April 14 Findings

## `libero10_logic_lora_from_libero_ckpt_20260416_014018`

The name `libero10_logic_lora_from_libero_ckpt_20260416_014018` corresponds to a LoRA training run, not to a completed eval run.

### Corrected What Happened

- The matching Slurm job is `445175`.
- W&B metadata, the Slurm log, and Slurm accounting all agree that it really ran on `dj-l40-0.grasp.maas`.
- It was not a Python-crash failure in training code.
- It also was not just stuck after dataset fetch. The log shows it progressed through:
  - W&B init
  - norm-stats load
  - dataset file fetch
  - data-loader initialization
  - parameter sharding
  - checkpoint restore start from `/home/chriswatson/.cache/openpi/openpi-assets/checkpoints/pi0_fast_libero/params`
- After that progress, Slurm sent `TERM`, the wrapper trapped it, and the job requested a requeue.

### Exact Lifecycle

- Slurm accounting records the run starting on `2026-04-16 01:40:55` and ending in `REQUEUED` state on `2026-04-16 01:46:34`.
- The wrapper log timestamps are four hours ahead of the Slurm accounting timestamps and appear to be UTC-like:
  - `2026-04-16 05:40:17` host banner on `dj-l40-0.grasp.maas`
  - `2026-04-16 05:43:28` data loader initialized
  - `2026-04-16 05:43:29` checkpoint restore started
  - `2026-04-16 05:45:55` Slurm cancellation line
  - `2026-04-16 05:45:56` wrapper caught `TERM`
  - `2026-04-16 05:45:58` wrapper requested requeue
- Slurm then created a requeued pending record for the same job ID, but that follow-up attempt never got a node and was later cancelled before starting.

### Why an Eval Would Fail or Be Unavailable

- There was no completed checkpoint to evaluate.
- The checkpoint directory exists, but it only contains `wandb_id.txt` and no checkpoint step directories.
- I did not find a matching eval output directory under `data/libero/evals/`.

### Interpretation

- `dj-l40-0.grasp.maas` is not sufficient to explain the failure by itself.
- A known-good full fine-tune, job `444316`, completed successfully on the same node.
- The strongest current reading is:
  - `445175` genuinely ran on `dj-l40-0.grasp.maas`
  - it made it past dataset setup
  - it was then stopped externally by Slurm or a user action before the first checkpoint save
- So the absence of checkpoints for this run is explained by external termination, not by proof of a node-specific startup hang.

### Evidence

- Training log: [logs/slurm/openpi_libero10_logic_lora-445175.out](/home/chriswatson/openpi-finetune/openpi/logs/slurm/openpi_libero10_logic_lora-445175.out:1)
- W&B metadata: [wandb/run-20260416_054035-8ye7tzo7/files/wandb-metadata.json](/home/chriswatson/openpi-finetune/openpi/wandb/run-20260416_054035-8ye7tzo7/files/wandb-metadata.json:1)
- Empty checkpoint root: [checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_from_libero_ckpt_20260416_014018](/home/chriswatson/openpi-finetune/openpi/checkpoints/pi0_fast_libero_10_logic_low_mem_finetune/libero10_logic_lora_from_libero_ckpt_20260416_014018:1)

## Nearby But Different Failure

There was a different LoRA experiment with a very similar timestamp:

- `libero10_logic_lora_20260416_014018`
- Slurm job `445176`

That one did fail with a real training error, also on `dj-l40-0.grasp.maas`.

### What Failed There

- It started at essentially the same time as `445175` on the same node.
- W&B metadata shows it used the same host, `dj-l40-0.grasp.maas`.
- It failed during dataset metadata loading with:
  `FileNotFoundError: [Errno 2] No such file or directory: '/home/chriswatson/.cache/huggingface/lerobot/physical-intelligence/libero/meta/info.json'`

### Why This Matters

- The two runs should not be conflated.
- `445175`:
  external termination after progressing into checkpoint restore
- `445176`:
  concrete dataset metadata failure from the shared Hugging Face LeRobot cache path
- Because both jobs were running concurrently on the same node, the shared-cache race remains a plausible explanation for `445176`.
- But the evidence for `445175` is different: it got past dataset initialization and then was terminated.

### Evidence

- Failure log: [logs/slurm/openpi_libero10_logic_lora-445176.out](/home/chriswatson/openpi-finetune/openpi/logs/slurm/openpi_libero10_logic_lora-445176.out:1)
- W&B metadata: [wandb/run-20260416_054034-strsoqar/files/wandb-metadata.json](/home/chriswatson/openpi-finetune/openpi/wandb/run-20260416_054034-strsoqar/files/wandb-metadata.json:1)

