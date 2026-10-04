# Tracking API

`LocalTracker` writes versioned directories with one CSV row per epoch and the latest
metric values in the YAML summary. `WandbTracker` retains step-level logging.

::: torch_batteries.tracking
