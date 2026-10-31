# Installation

## Requirements

`torch-batteries` requires Python 3.12 or newer and PyTorch 2.9 or newer. The core
package supports CPU, CUDA, and Apple MPS devices. Install it from PyPI:

```bash
python -m pip install torch-batteries
```

The core installation includes PyTorch and `tqdm`. It does not install notebook,
plotting, vision, reinforcement-learning, diffusion, or tracking packages.

## Compatibility checks

The native CPU compatibility CI targets these combinations:

| Platform | Architecture | Python versions |
| --- | --- | --- |
| Ubuntu | x64 | 3.12, 3.13, 3.14 |
| macOS | Apple Silicon / ARM64 | 3.12, 3.13, 3.14 |
| Windows | x64 | 3.12, 3.13, 3.14 |
| Windows 11 | ARM64 | 3.12, 3.13 |

Each job builds and installs a wheel, checks dependencies, runs the runtime suite,
and exercises synthetic training, evaluation, prediction, checkpoint resume, and a
spawned DataLoader worker. This tests the selected CPU dependency versions; CUDA,
MPS, optional integrations, and every permitted dependency version require additional
validation. Native Inductor compilation is checked separately in Ubuntu quality CI.

Intel macOS is excluded because official PyTorch x64 binaries stopped after 2.2,
below this package's PyTorch minimum. See the
[upstream announcement](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).
Windows ARM/Python 3.14 is excluded while a matching CPU wheel is unavailable in the
[official PyTorch CPU index](https://download.pytorch.org/whl/cpu/torch/).
For Windows ARM, install the native CPU build first:

```bash
python -m pip install "torch>=2.9.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-batteries
```

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
