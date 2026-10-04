# Release Notes

## 1.0.0 — 2026-09-20

- Added named training and validation datasets with deterministic round-robin or
  seeded interleaved batches, sample-weighted losses and metrics, and dataset-specific
  metric names for multi-dataset workflows.
- Added phase-specific metric configuration and `MetricSpec` selectors for structured
  model outputs and targets.
- Added validation cadence, cumulative epoch and optimizer-step counts, and explicit
  stop reasons to training results.
- Added `LocalTracker` with versioned run directories, epoch-level CSV metrics,
  hyperparameter YAML, and latest-value summaries.
- Updated the existing examples and added USPS/SEMEION, Ames Housing multi-target,
  and FrozenLake A2C notebooks.
- Removed pre-1.0 callback aliases and legacy full-checkpoint loading in
  [PR #25](https://github.com/michalszc/torch-batteries/pull/25).

## 0.12.0 — 2026-09-10

- Added compatibility contracts that detect removal or incompatible changes to the
  documented public API and verify that protected endpoints remain in the generated
  reference documentation.
- Verified optional `torch.compile` use across training, validation, testing,
  prediction, automatic metrics, optimizer parameters, and checkpoint restoration.
- Added strict-by-default metric lifecycle errors with the optional
  `metric_error_policy="warn"` policy, protected runtime configuration from mutation
  inside events, and made missing monitored metrics fail consistently.
- Added `TerminateOnNonFinite` for selected loss and metric checks before unsafe
  training operations and across evaluation phase results.
- Added `Event.ON_EXCEPTION` for escaping workflow failures and failure-aware
  experiment-tracker cleanup without final model artifact logging.
- Expanded full checkpoints with Python, PyTorch CPU, available CUDA and MPS,
  optional NumPy, and DataLoader generator RNG state.
- Made full and raw-model checkpoint restoration transactional, with complete
  callback-state preflight, component rollback, and preservation of the original
  load failure.
- Clarified scalar batch-mean loss aggregation, DataPack generator behavior, and the
  optional compile-before-`Battery` usage order in
  [PR #24](https://github.com/michalszc/torch-batteries/pull/24).

## 0.11.0 — 2026-08-17

- Added `Battery.fit()` for training with optional per-epoch validation and
  `Battery.validate()` for a required, optimizer-free standalone validation pass.
  Added the public `FitResult` and `ValidationResult` contracts.
- Retained validation through `Battery.train()` for compatibility while deprecating
  its `val_loader` parameter and `TrainResult.val_loss`/`val_metrics` fields. The
  compatibility path now logs and emits a deprecation warning only when validation
  actually runs.
- Unified charged-method discovery and dispatch behind shared handler infrastructure
  while keeping the full public `EventHandler` and `DataPackHandler` APIs documented.
- Split production classes, protocols, enums, and helper functions into focused
  modules and natural data-types, event-core, trainer-types, progress-types, W&B,
  logging, and metrics packages without changing existing public import paths.
- Split Battery checkpointing, training, evaluation, prediction, and shared workflow
  state into focused private modules while retaining checkpoint compatibility; made
  `@charge` stackable with deterministic duplicate and competing-handler validation.
- Updated guides, API references, executable documentation examples, and all seven
  notebooks for `fit()`, standalone validation, result contracts, canonical
  `phase="validation"`, and the reorganized public API.
- Aligned exception ownership, built-in error types, module-qualified logging,
  lifecycle diagnostics, docstring coverage, and notebook-focused CI validation in
  [PR #23](https://github.com/michalszc/torch-batteries/pull/23).

## 0.10.0 — 2026-08-11

- Renamed callback metric-monitoring configuration from `stage` to `phase` in
  `EarlyStopping`, `ModelCheckpoint`, and `LearningRateScheduler`. The deprecated
  `stage=` keyword remains available as a compatibility alias.
- Added event-driven `DataPack` workflows for reusable dataset and DataLoader
  construction across training, testing, and prediction while preserving the direct
  DataLoader API.
- Added validated phase-aware `DataLoaderConfig` materialization, deterministic
  opt-in generators, guaranteed teardown, and DataPack state in full checkpoints.
- Added context-managed `DataPack.resolve()` for constructing and inspecting datasets
  and DataLoaders without a `Battery`, backed by the same lifecycle resolver used by
  implicit Battery workflows.
- Added the DataPack guide and API reference, converted the MNIST notebook to the
  implicit DataPack workflow, and retained direct DataLoader coverage in the
  function-fitting notebook in
  [PR #22](https://github.com/michalszc/torch-batteries/pull/22).

## 0.9.0 — 2026-08-08

- Replaced pdoc with a structured MkDocs Material site, clarified public contracts,
  updated all example notebooks, and added metadata-free static documentation and
  notebook validation, and adopted the Apache License 2.0 without changing runtime
  API behavior in
  [PR #21](https://github.com/michalszc/torch-batteries/pull/21).

## 0.8.0 — 2026-08-02

- Added event-driven optimization controls, callback state restoration, stateful
  metrics, prediction streaming and recursive aggregation, resumable training, and
  expanded user-facing examples in
  [PR #20](https://github.com/michalszc/torch-batteries/pull/20).

## 0.7.0 — 2026-07-16

- Introduced the explicit `StepOutput` contract and hardened workflow validation,
  callback snapshots, checkpoint storage, logging, and regression coverage in
  [PR #19](https://github.com/michalszc/torch-batteries/pull/19).

## 0.6.0 — 2026-06-11

- Added explicit history data to training lifecycle event contexts and aligned the
  trainer, events, tests, examples, and public types in
  [PR #18](https://github.com/michalszc/torch-batteries/pull/18).

## 0.5.3 — 2026-06-11

- Made W&B a genuinely optional runtime dependency and improved related device,
  tracking, and checkpoint coverage in
  [PR #17](https://github.com/michalszc/torch-batteries/pull/17).

## 0.5.2 — 2026-01-19

- Expanded the image-classification notebook and example dependency set in
  [PR #12](https://github.com/michalszc/torch-batteries/pull/12).
- Subsequent repository maintenance at version 0.5.2 refined dependency groups,
  workflow triggers, and release automation in
  [PR #15](https://github.com/michalszc/torch-batteries/pull/15), updated contributor
  guidance in [PR #13](https://github.com/michalszc/torch-batteries/pull/13), and added
  a documentation badge in
  [PR #16](https://github.com/michalszc/torch-batteries/pull/16).

## 0.5.1 — 2026-01-18

- Corrected documentation installation for the W&B-enabled package in
  [PR #11](https://github.com/michalszc/torch-batteries/pull/11).

## 0.5.0 — 2026-01-18

- Added backend-neutral experiment tracking, the W&B integration and callback, model
  artifact logging, and a learning-rate sweep notebook in
  [PR #9](https://github.com/michalszc/torch-batteries/pull/9).

## 0.4.2 — 2026-01-18

- Fixed CI workflow behavior and expanded module docstrings and API examples in
  [PR #10](https://github.com/michalszc/torch-batteries/pull/10).

## 0.4.1 — 2026-01-17

- Added the MNIST image-classification notebook demonstrating callbacks in
  [PR #8](https://github.com/michalszc/torch-batteries/pull/8).

## 0.4.0 — 2026-01-17

- Added callback support, early stopping, model checkpoints, and their initial test
  coverage in [PR #7](https://github.com/michalszc/torch-batteries/pull/7).

## 0.3.0 — 2025-12-31

- Added metric collection and calculation, richer `EventContext` contracts, and
  corresponding trainer and utility coverage in
  [PR #6](https://github.com/michalszc/torch-batteries/pull/6).

## 0.2.1 — 2025-12-14

- Reworked progress reporting into silent, simple, and progress-bar implementations
  behind `ProgressFactory`, and reorganized the test suite in
  [PR #5](https://github.com/michalszc/torch-batteries/pull/5).

## 0.2.0 — 2025-12-05

- Introduced `Battery`, event charging and dispatch, workflow result types, automatic
  device and batch handling, progress reporting, package typing, and the first
  end-to-end example in
  [PR #4](https://github.com/michalszc/torch-batteries/pull/4).

## 0.1.0 — 2025-11-28

- Established the package structure, development tooling, CI/CD, publishing,
  version validation, contribution guidance, and initial example in
  [PR #1](https://github.com/michalszc/torch-batteries/pull/1).
- Consolidated CI checks into a reusable helper in
  [PR #2](https://github.com/michalszc/torch-batteries/pull/2).
- Added the original pdoc documentation build and GitHub Pages deployment in
  [PR #3](https://github.com/michalszc/torch-batteries/pull/3).
