# Trainer API

`Battery` remains the public workflow facade. Its checkpoint, fitting, training,
standalone validation, testing, prediction, and streaming-prediction methods are
documented here even though their
implementations are organized into focused private trainer modules. Applications
should continue importing only `Battery` from `torch_batteries` or
`torch_batteries.trainer`.

Use `fit()` for combined training and optional validation, `train()` for
training-only work, and `validate()` for a required single validation pass. `fit()`
can validate on absolute epoch multiples through `validate_every_n_epochs`. Train and
fit results include cumulative completion counters and a stop reason. Use
`request_stop(reason)` from callbacks for an orderly stop.

::: torch_batteries.trainer
