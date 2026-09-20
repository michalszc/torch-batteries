# Metrics API

`MetricSpec` selects top-level prediction and target keys from structured
`StepOutput` values. A flat metric mapping applies to all phases; nested mappings
configure train, validation, and test separately.

::: torch_batteries.utils.metrics
