# Repository Guidelines

## What This Repo Is
`openpi` is a Python training and serving repo for OpenPI policies. Most active work happens in:

- `src/openpi/`: training, policies, models, transforms, serving
- `scripts/`: entrypoints like training and policy serving
- `examples/libero/`: LIBERO-specific training and eval tooling
- `packages/openpi-client/`: local client package used by eval and serving flows

Treat `third_party/` as vendored code. Avoid editing it unless the task explicitly requires it.

## Important Paths
- `src/openpi/training/config.py`: train config definitions and named configs
- `src/openpi/training/data_loader.py`: dataset wiring, including LIBERO prompt selection
- `src/openpi/serving/websocket_policy_server.py`: websocket serving path
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py`: websocket client used by eval
- `scripts/train.py`: main training entrypoint
- `scripts/serve_policy.py`: local/server policy launcher
- `examples/libero/main.py`: LIBERO eval driver
- `examples/libero/train_libero_10_logic_full.slurm`: current Slurm training launcher for logic-prompt LIBERO runs
- `examples/libero/eval_checkpoint.slurm`: current Slurm eval job
- `scripts/submit_libero_eval_slurm.sh`: helper for submitting eval jobs
- `docs/libero_eval_slurm.md`: human-facing instructions for the eval workflow
- `EXPERIMENTS_APRIL_14.md`: concrete notes from the April 14 LIBERO training and eval session

## Environment And Tooling
Use Python 3.11+ for the main repo and prefer `uv`.

Common commands:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format .
uv run scripts/train.py ...
uv run scripts/serve_policy.py --port=8000 policy:checkpoint ...
```

The root project is the primary dev environment. LIBERO eval is different:

- the main repo uses Python 3.11
- LIBERO eval currently uses a separate Python 3.10 environment
- the Slurm eval flow can build a job-local env under `/tmp/...`
- do not assume the root `.venv` is enough for LIBERO simulator work

Follow `docs/libero_eval_slurm.md` and the scripts in `examples/libero/` for LIBERO eval instead of inventing a new bootstrap path.

## Repo-Specific Conventions
Use 4-space indentation, type hints, and Python 3.11 syntax. Ruff enforces formatting and import sorting. Keep new code in `src/openpi/...` unless it is clearly a script, example, or package-level change.

Naming:

- functions, modules, variables: `snake_case`
- classes: `PascalCase`
- configs: descriptive names like `pi0_fast_libero_10_logic`

When editing markdown or docs, keep links repo-portable. Do not commit local absolute `/home/...` paths.

## Training And Eval Notes
Current LIBERO work in this repo uses two prompt styles:

- default English dataset task descriptions
- logic prompt overrides from `data/libero/libero_10_logic_descriptions.json`

Be precise about where prompt behavior comes from:

- training-time LIBERO prompt semantics come from the selected config, especially `task_description_path`, not from the experiment name or checkpoint folder name
- eval-time prompt semantics currently include one naming heuristic: `scripts/submit_libero_eval_slurm.sh` and `examples/libero/eval_checkpoint.slurm` auto-set `PROMPT_OVERRIDE_FILE=data/libero/libero_10_logic_descriptions.json` when `POLICY_CONFIG` contains `_logic`
- do not extend that heuristic to `EXP_NAME`, checkpoint leaf names, or other filenames; if behavior matters, pass `--prompt-override-file` explicitly or wire it through config/CLI fields with explicit semantics

If you touch LIBERO prompt plumbing, make sure task filtering still works correctly with the logic description file. The relevant regression coverage lives in `src/openpi/training/data_loader_test.py`.

W&B support is wired through both training and eval:

- training supports config-level `wandb_tags` and `wandb_group`
- training Slurm wrappers may pass `OPENPI_WANDB_TAGS` and `OPENPI_WANDB_GROUP`
- eval uses explicit CLI flags in `examples/libero/main.py` and the Slurm wrappers

Do not silently remove or bypass this metadata plumbing when changing launch scripts.

The websocket serving path has already needed one important fix: policy inference must not block keepalive handling for long JAX/XLA steps. Be careful when changing either side of the websocket boundary:

- `src/openpi/serving/websocket_policy_server.py`
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py`
- `examples/libero/main.py`

## Testing Guidance
Run targeted tests first, then broader checks if practical.

Examples:

```bash
uv run pytest src/openpi/training/data_loader_test.py
uv run ruff check src/openpi scripts packages
bash -n examples/libero/eval_checkpoint.slurm scripts/submit_libero_eval_slurm.sh
```

Some tests depend on local datasets, GPU libraries, simulators, or external env setup. If you cannot run something, say exactly what blocked verification.

## Data And Artifact Hygiene
Do not commit large datasets, checkpoints, generated eval outputs, or secrets.

Usually untracked/generated:

- `checkpoints/`
- `logs/`
- `data/libero/evals/`
- ad hoc local envs or temp files

Tracked exceptions are allowed when they are deliberate repo inputs, such as `data/libero/libero_10_logic_descriptions.json`.

Keep experiment notes in tracked markdown like `EXPERIMENTS_APRIL_14.md` or under `docs/`, not in shell history or disposable scratch files.

## Commits And PRs
Keep commits focused and use short imperative subjects. Small stacked commits are preferred over one mixed commit when the changes split cleanly by concern.

If behavior changes, include exact verification steps in the PR or handoff note. If you update scripts or docs that define a workflow, keep them in sync in the same change.
