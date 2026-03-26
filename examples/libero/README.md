# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

export SERVER_ARGS="--env LIBERO"
docker compose -f examples/libero/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
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
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src:$PWD/third_party/libero
export LIBERO_CONFIG_PATH=$PWD/.cache/libero-openpi

# Run the simulation
python examples/libero/main.py --args.host 127.0.0.1 --args.port 8000
```

Terminal window 2:

```bash
# Run the server
USE_TF=0 uv run scripts/serve_policy.py --port=8000 --env LIBERO
```

Use Python 3.10 for `examples/libero/.venv`. Python 3.8 is too old for the current `examples/libero/main.py`, and Python 3.11 does not have a matching `torch==1.11.0+cu113` wheel.

## Results

If you follow the training instructions and hyperparameters in the `pi0_libero` and `pi0_fast_libero` configs, you should get results similar to the following:

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0-FAST @ 30k (finetuned) | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π0 @ 30k (finetuned) | 96.8 | 98.8 | 95.8 | 85.2 | 94.15 |

Note that the hyperparameters for these runs are not tuned and $\pi_0$-FAST does not use a FAST tokenizer optimized for Libero. Likely, the results could be improved with more tuning, we mainly use these results as an example of how to use openpi to fine-tune $\pi_0$ models on a new dataset.
