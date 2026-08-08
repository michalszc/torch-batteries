# Release Notes

## 0.9.0 — 2026-08-08

- Replaced pdoc with a structured MkDocs Material site, clarified public contracts,
  updated all example notebooks, and added metadata-free static documentation and
  notebook validation without changing runtime API behavior in
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
