# Repository Guidelines

## Project Structure & Module Organization
Core Python code lives in [`src/openpi`](./src/openpi), with training, policies, models, and transforms under that tree. Entry-point scripts are in [`scripts`](./scripts). End-to-end examples and environment-specific utilities live in [`examples`](./examples), including LIBERO workflows in [`examples/libero`](./examples/libero). The local client package is in [`packages/openpi-client`](./packages/openpi-client). Assets, downloaded data, checkpoints, and run logs typically live in [`assets`](./assets), [`data`](./data), [`checkpoints`](./checkpoints), and [`logs`](./logs). Third-party code is vendored in [`third_party`](./third_party) and should usually not be edited.

## Build, Test, and Development Commands
Use `uv` with Python 3.11+ for main development.

```bash
uv sync --dev                 # install project + dev tools
uv run pytest                 # run test suite
uv run ruff check .           # lint
uv run ruff format .          # format
uv run scripts/train.py ...   # launch training
uv run scripts/serve_policy.py --port=8000 policy:checkpoint ...
```

For LIBERO eval, the repo currently uses a separate `examples/libero/.venv` due simulator and Torch constraints; follow the example docs instead of assuming the root env is sufficient.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and Python 3.11 syntax. Ruff enforces import sorting, formatting, and many lint rules; line length is 120. Prefer `snake_case` for functions, variables, and module names, `PascalCase` for classes, and descriptive config names such as `pi0_fast_libero_object_train7_low_mem_finetune`. Keep new code under `src/openpi/...` unless it is a top-level script or example.

## Testing Guidelines
Tests run with `pytest`; discovery is configured for `src`, `scripts`, and `packages`. Name tests `*_test.py` and keep them close to the code they cover. Run targeted tests while iterating, then finish with `uv run pytest`. Some tests or repo fixtures require extra system dependencies or GPU libraries; document any gaps if you cannot run them.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `Fix typo in DROID README.md` or `add UR5 example`. Keep commits focused and avoid bundling unrelated changes. PRs should include a clear description, linked issue or discussion when relevant, and exact verification steps. If behavior changes in examples or docs, update the corresponding README in the same PR.

## Configuration & Data Notes
Do not commit large datasets, checkpoints, or secrets. Training outputs are stored under `checkpoints/<config-name>/<exp-name>`. LIBERO raw data and derived artifacts belong under `data/libero/`, and local experiment notes belong in docs or tracked markdown, not ad hoc shell history.
