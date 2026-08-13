# DataPack Workflows

`DataPack` is the optional high-level data boundary for a `Battery`. It keeps dataset
preparation, split construction, and DataLoader policy together while leaving dataset
and transform implementations as ordinary PyTorch code.

Use a DataPack when several workflows share the same data definition or when data
construction has state that must survive a checkpoint. Continue passing explicit
DataLoaders when the caller already owns them or needs one-off loader composition.

## Define the lifecycle

```python
import torch
from torch.utils.data import TensorDataset

from torch_batteries import (
    DataContext,
    DataLoaderConfig,
    DataPack,
    DatasetBundle,
    Event,
    charge,
)


class RegressionData(DataPack):
    seed = 7

    @charge(Event.PREPARE_DATA)
    def prepare(self, context: DataContext) -> None:
        # Download or populate a cache here. This runs once per Battery.
        pass

    @charge(Event.SETUP_DATA)
    def setup(self, context: DataContext) -> DatasetBundle:
        generator = context["generator"]
        inputs = torch.randn(96, 4, generator=generator)
        targets = inputs.sum(dim=1, keepdim=True)
        dataset = TensorDataset(inputs, targets)
        train, validation, test = torch.utils.data.random_split(
            dataset,
            [64, 16, 16],
            generator=generator,
        )
        return DatasetBundle(
            train=train,
            validation=validation,
            test=test,
            predict=test,
        )

    @charge(Event.CONFIGURE_DATALOADER)
    def loader(self, context: DataContext) -> DataLoaderConfig:
        return DataLoaderConfig(batch_size=16)

    @charge(Event.TEARDOWN_DATA)
    def teardown(self, context: DataContext) -> None:
        # Close workflow-scoped files or connections here.
        pass
```

Attach it once and omit loaders from data-backed workflows:

```python
battery = Battery(model, optimizer=optimizer, data_pack=RegressionData())
battery.train(epochs=10)
battery.test()
battery.predict(move_to_cpu=True, concatenate=True)
```

Test and prediction phases can expose several named datasets:

```python
return DatasetBundle(
    train=train,
    validation=validation,
    test={"Test1": test_1, "Test2": test_2},
    predict={"Predict1": predict_1, "Predict2": predict_2},
)
```

Without a selector, `battery.test()` and `battery.predict()` run every named dataset
and return results keyed by those names. Select one dataset when only one pass is
needed:

```python
test_2_result = battery.test(dataset="Test2")
predict_1_result = battery.predict(dataset="Predict1")
```

A singular dataset, or one selected by name, retains the ordinary singular result
shape. Training and validation datasets remain singular.

`datasets_for_phase()` normalizes a singular dataset under the name `"default"` so
internal workflow code can handle singular and named datasets uniformly. Therefore,
`dataset="default"` selects a singular test or prediction dataset, although omitting
the selector has the same result. Named mappings should use meaningful domain names
instead of relying on `"default"`.

`PREPARE_DATA` is for idempotent downloads and cache population. It runs at most once
for an attached DataPack. `SETUP_DATA` runs once for each `train`, `test`, or
`predict` call. `CONFIGURE_DATALOADER` runs for every dataset used by that call.
`TEARDOWN_DATA` always runs after an implicit workflow, including when setup, loader
construction, or model execution raises.

## Understand the context

`DataContext` contains `battery`, `data_pack`, `stage`, and `device`. Setup also gets
an optional `seed` and `generator`; loader configuration additionally gets `phase`,
`datasets`, the current `dataset`, and its `dataset_name`. Test and prediction event
contexts expose the same identity field. Teardown receives `datasets` when setup
completed.

`dataset_name` is the stable identifier intended for logging and branching.

There is no framework default seed. Define a non-negative integer `seed` attribute on
the DataPack only when its construction needs deterministic generators. Each phase
gets a separate generator derived from that seed. All datasets in the same phase
receive the same phase seed. A `DataLoaderConfig.generator` overrides the context
generator.

## Configure DataLoaders

`DataLoaderConfig` mirrors the common PyTorch DataLoader options and validates
incompatible combinations before a workflow starts. Its phase-aware defaults are:

- Map-style training datasets shuffle automatically.
- Validation, test, prediction, and iterable datasets do not shuffle.
- `pin_memory="auto"` enables pinning only for a CUDA Battery.
- `num_workers=0` avoids worker processes; prefetching and persistent workers require
  a positive worker count.

Explicit sampler and batch-sampler rules match PyTorch. A sampler cannot accompany
`shuffle=True`. A batch sampler requires `batch_size=None` and cannot accompany
shuffle, a sampler, or `drop_last`.

Return an existing `torch.utils.data.DataLoader` from `CONFIGURE_DATALOADER` when the
high-level configuration cannot express a custom loader. That loader must still be
sized and non-empty because Battery's progress and aggregation contracts require
`len(loader)`.

## Choose explicit or implicit mode

An explicit primary loader selects direct-loader mode for the whole invocation:

```python
battery.train(custom_train_loader, custom_validation_loader)
```

Battery does not silently combine that train loader with validation data from the
DataPack. Passing only an explicit validation loader is therefore invalid. The same
rule keeps testing and prediction unambiguous: either pass their primary loader or
omit it and use the DataPack.

Validation data is optional during implicit training. Train, test, and prediction
datasets become required only when their corresponding implicit workflow is called.

## Preserve data construction state

Override `state_dict()` and `load_state_dict()` for values that affect later setup,
such as stored split indices or a streaming cursor:

```python
class SplitData(DataPack):
    def state_dict(self) -> dict[str, object]:
        return {"split_indices": self.split_indices}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.split_indices = state_dict["split_indices"]
```

Full checkpoints store this dictionary and the DataPack's qualified type. During
`train(resume_from=...)`, it is restored before `SETUP_DATA`. Datasets, DataLoaders,
worker processes, and open resources are never serialized. A resume requires the
same DataPack type to be attached; checkpoints created before DataPack state was
introduced remain readable.

Distributed sampling and unsized streaming loaders are outside the current contract.
Configure ordinary sized PyTorch loaders and manage distributed sampling in
application code for now.
