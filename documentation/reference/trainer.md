# Trainer API

`Battery` remains the public workflow facade. Its checkpoint, training, testing,
prediction, and streaming-prediction methods are documented here even though their
implementations are organized into focused private trainer modules. Applications
should continue importing only `Battery` from `torch_batteries` or
`torch_batteries.trainer`.

::: torch_batteries.trainer
