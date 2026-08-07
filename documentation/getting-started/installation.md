# Installation

## Requirements

`torch-batteries` requires Python 3.12 or newer and PyTorch 2.9 or newer. The core
package supports CPU, CUDA, and Apple MPS devices. Install it from PyPI:

```bash
python -m pip install torch-batteries
```

The core installation includes PyTorch and `tqdm`. It does not install notebook,
plotting, vision, reinforcement-learning, diffusion, or tracking packages.

## Optional dependencies

Install W&B only when experiment tracking is needed:

```bash
python -m pip install "torch-batteries[wandb]"
```

Install the dependencies used by the maintained notebooks:

```bash
python -m pip install "torch-batteries[example]"
```

Install both groups when reproducing every example:

```bash
python -m pip install "torch-batteries[all]"
```

## Device selection

`Battery` uses `device="auto"` by default and chooses CUDA, then MPS, then CPU.
Pass an explicit device when placement must be controlled:

```python
battery = Battery(model, optimizer=optimizer, device="cuda:0")
```

The model is moved during `Battery` construction. Loader batches are moved
recursively before each step. Non-tensor metadata is preserved.

## Confirm the installation

```bash
python -c "import torch_batteries; print(torch_batteries.__version__)"
```

If `WandbTracker` reports that W&B is unavailable, install the `wandb` extra in the
same environment that runs the training process.
