# Examples

The maintained notebooks progress from a small synthetic regression problem to
complete computer-vision, reinforcement-learning, and experiment-tracking workflows.
Each notebook includes an installation cell, reports the installed torch-batteries
version, and can be opened directly in Google Colab.

| Notebook | What it demonstrates | Expected workload | Links |
| --- | --- | --- | --- |
| **Function Fitting with MLP** | Training, validation, testing, prediction, metrics, and result inspection on synthetic data | CPU-friendly; no downloads; normally under a minute | [![View on GitHub](https://img.shields.io/badge/View_on-GitHub-181717?logo=github)](https://github.com/michalszc/torch-batteries/blob/master/notebooks/function_fitting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/michalszc/torch-batteries/blob/master/notebooks/function_fitting.ipynb) |
| **Image Classification with CNN** | MNIST DataPack lifecycle, implicit loaders, early stopping, and model checkpoints | Downloads MNIST; accelerator recommended | [![View on GitHub](https://img.shields.io/badge/View_on-GitHub-181717?logo=github)](https://github.com/michalszc/torch-batteries/blob/master/notebooks/image_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/michalszc/torch-batteries/blob/master/notebooks/image_classification.ipynb) |
| **Learning Rate Sweep with Early Stopping** | Comparing learning rates, eager early stopping, and offline W&B experiment tracking | Downloads MNIST; trains five models; accelerator recommended | [![View on GitHub](https://img.shields.io/badge/View_on-GitHub-181717?logo=github)](https://github.com/michalszc/torch-batteries/blob/master/notebooks/lr_sweep_early_stopping.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/michalszc/torch-batteries/blob/master/notebooks/lr_sweep_early_stopping.ipynb) |
| **FashionMNIST Diffusion** | Gradient accumulation, mixed precision, clipping, scheduling, and streaming generation | Downloads FashionMNIST; CUDA or MPS strongly recommended | [![View on GitHub](https://img.shields.io/badge/View_on-GitHub-181717?logo=github)](https://github.com/michalszc/torch-batteries/blob/master/notebooks/fashion_mnist_diffusion.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/michalszc/torch-batteries/blob/master/notebooks/fashion_mnist_diffusion.ipynb) |
| **CartPole Reinforcement Learning** | DQN training, optimization events, stateful metrics, scheduling, and `predict_iter()` | CPU-compatible; no dataset or credentials | [![View on GitHub](https://img.shields.io/badge/View_on-GitHub-181717?logo=github)](https://github.com/michalszc/torch-batteries/blob/master/notebooks/cartpole_reinforcement_learning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/michalszc/torch-batteries/blob/master/notebooks/cartpole_reinforcement_learning.ipynb) |
| **CIFAR-10 Transfer Learning** | Resumable callbacks, metric types, top-k checkpoints, save/load, resume modes, and structured prediction | Downloads CIFAR-10 and ResNet18 weights; accelerator recommended | [![View on GitHub](https://img.shields.io/badge/View_on-GitHub-181717?logo=github)](https://github.com/michalszc/torch-batteries/blob/master/notebooks/cifar10_transfer_learning.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/michalszc/torch-batteries/blob/master/notebooks/cifar10_transfer_learning.ipynb) |

## Where to start

Start with **Function Fitting with MLP** for the shortest direct-DataLoader
introduction. Continue with **Image Classification with CNN** for DataPack, callbacks,
and checkpoints, then
choose a specialized notebook according to the feature you want to explore.

The notebooks install their own example dependencies when run in a fresh notebook
environment. Some datasets and pretrained weights are downloaded on first use; each
notebook describes its prerequisites and expected hardware before running any
training code.
