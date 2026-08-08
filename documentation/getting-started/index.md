# Getting Started

This section takes you from installation to a complete event-driven PyTorch workflow.
By the end, you will know where training behavior lives, what `Battery` manages, and
how to inspect the results returned by each workflow.

## Before you begin

You should be comfortable creating a PyTorch `nn.Module`, optimizer, dataset, and
`DataLoader`. No knowledge of the torch-batteries event system is required. The first
example runs on CPU with synthetic data and does not need downloads or external
services.

## Recommended path

| Step | Page | What you will accomplish |
| --- | --- | --- |
| 1 | [Installation](installation.md) | Install the core package or an optional extra and verify the active version. |
| 2 | [Quick Start](quickstart.md) | Train, validate, test, and predict with a complete copyable example. |
| 3 | [Core Concepts](core-concepts.md) | Understand the roles of the model, events, `Battery`, metrics, callbacks, and results. |

Read these pages in order on your first visit. Later, the Core Concepts page is a
useful map when deciding whether new behavior belongs in a charged model method, a
callback, or application code.

## What the quick start establishes

A standard torch-batteries workflow has a small public surface:

1. A model owns forward computation and methods charged to lifecycle events.
2. Step methods return `StepOutput` with loss, predictions, targets, and optional
   manual metrics.
3. `Battery` owns device placement, phase loops, optimization, metric aggregation,
   callbacks, and structured results.
4. The application creates loaders and decides how to persist or consume results.

## Continue by task

After completing the quick start, choose the guide that matches your next problem:

- [Training and Evaluation](../guides/training.md) for workflow contracts and legacy
  step-return forms.
- [Metrics](../guides/metrics.md) for callable, stateful, or collected calculations.
- [Batches and Devices](../guides/batches-and-devices.md) for nested inputs and device
  movement.
- [Callbacks and Optimization](../guides/callbacks.md) for early stopping, schedulers,
  precision, clipping, and accumulation.
- [Checkpoints and Resume](../guides/checkpoints.md) for durable training state.
- [Prediction](../guides/prediction.md) for concatenated or streaming inference.

If something fails during setup, start with [Troubleshooting](../troubleshooting.md).
