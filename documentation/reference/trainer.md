# Trainer API

`Battery` remains the public workflow facade. Its checkpoint, fitting, training,
standalone validation, testing, prediction, and streaming-prediction methods are
documented here even though their
implementations are organized into focused private trainer modules. Applications
should continue importing only `Battery` from `torch_batteries` or
`torch_batteries.trainer`.

Use `fit()` for combined training and optional validation, `train()` for
training-only work, and `validate()` for a required single validation pass.

::: torch_batteries.trainer
