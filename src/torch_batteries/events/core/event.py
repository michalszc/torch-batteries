"""Core events and decorators for torch-batteries."""

from enum import Enum


class Event(Enum):
    """Events that can be used with the @charge decorator.

    Events are triggered at different points during training/testing/prediction.
    Model and callback events receive an `EventContext`; DataPack lifecycle events
    receive a `DataContext`. Whenever an event lists `epoch` in its context, the
    value follows the one-based public convention documented by `EventContext`.

    ## DataPack Lifecycle Events

    These events may be handled only by the DataPack attached to a Battery. The
    `stage` field is `"fit"`, `"test"`, or `"predict"`. A DataPack that defines a
    non-negative `seed` also receives `seed` and a deterministic `generator`.

    - `PREPARE_DATA`: Broadcast side-effect event for idempotent downloads and
      cache population. It runs at most once per Battery, before the first implicit
      setup.
        - **Context**: `battery`, `data_pack`, `stage`, `device`; optional `seed`,
          `generator`
        - **Return**: `None`
        - **Default**: no operation

    - `SETUP_DATA`: Exclusive provider called once for every implicit fit, test, or
      prediction workflow. Use `stage` to construct and return only the datasets
      required by the active workflow.
        - **Context**: same as `PREPARE_DATA`
        - **Return**: `DatasetBundle`
        - **Default**: none; an implicit workflow requires exactly one valid bundle

    - `CONFIGURE_DATALOADER`: Exclusive provider called for every dataset selected
      from the setup bundle. The phase-specific `generator` uses a deterministic
      offset from the DataPack seed.
        - **Context**: `battery`, `data_pack`, `stage`, `device`, `phase`,
          `datasets`, `dataset`, `dataset_name`; optional `seed`, `generator`
        - **Return**: `DataLoaderConfig` or `torch.utils.data.DataLoader`
        - **Default**: `DataLoaderConfig()`

    - `TEARDOWN_DATA`: Broadcast side-effect event that always runs when an implicit
      workflow exits, including after setup, loader, or model failures. `datasets`
      is present only when setup completed successfully.
        - **Context**: `battery`, `data_pack`, `stage`, `device`; optional `seed`,
          `generator`, `datasets`
        - **Return**: `None`
        - **Default**: no operation

    ## Optimization Extension Events

    The events below are public extension points. They may be handled by a
    method on the model or by a callback. Broadcast handlers run model-first
    and then in callback-list order. Exclusive providers and executors allow
    only one handler across the model and callbacks; discovery fails with a
    clear conflict error when more than one is registered.

    - `SETUP`: Broadcast once after Battery and event discovery are complete,
      before checkpoint state can be restored.
        - **Context**: `battery`, `model`, `optimizer`, `device`
        - **Return**: ignored
        - **Default**: no operation

    - `STEP_EXECUTION_CONTEXT`: Context-provider event requested immediately
      before `TRAIN_STEP`, `VALIDATION_STEP`, `TEST_STEP`, and `PREDICT_STEP`.
      Every handler must return a context manager. Context managers enter
      model-first and then in callback order, and always exit in reverse order,
      including when step execution raises.
        - **Context**: `battery`, `model`, `optimizer`, `device`, `phase`,
          `batch`, `batch_idx`, `epoch`
        - **Return**: a context manager
        - **Default**: `contextlib.nullcontext()`

    - `CONFIGURE_TRAIN_STEP`: Exclusive provider called after moving a training
      batch to the device and before zeroing gradients or running `TRAIN_STEP`.
        - **Context**: `battery`, `model`, `optimizer`, `device`, `phase`,
          `batch`, `batch_idx`, `total_batches`, `epoch`,
          `optimizer_step_idx`
        - **Return**: `OptimizationStep`
        - **Default**: `OptimizationStep()`

    - `BEFORE_BACKWARD`: Broadcast after parsing the training result and
      dividing its loss according to the optimization plan. Handlers may
      replace `backward_loss` but must not perform backward themselves.
        - **Context**: training batch fields plus `loss_tensor`,
          `backward_loss`, `optimization_plan`, `optimizer_step`, and
          `optimizer_step_idx`
        - **Return**: ignored

    - `BACKWARD`: Exclusive executor for backpropagation. It runs for every
      training batch, including intermediate accumulation batches.
        - **Context**: same as `BEFORE_BACKWARD`
        - **Return**: `None`
        - **Default**: `backward_loss.backward()`

    - `AFTER_BACKWARD`: Broadcast after backward succeeds. It is not emitted
      when backward raises. Gradients may still be AMP-scaled at this point.
        - **Context**: same as `BEFORE_BACKWARD`
        - **Return**: ignored

    - `BEFORE_GRADIENT_CLIP`: Broadcast only on a real optimizer boundary,
      after backward and before clipping. Mixed-precision handlers use it to
      unscale gradients. Every handler finishes before `GRADIENT_CLIP`.
        - **Context**: same as `BEFORE_BACKWARD`
        - **Return**: ignored

    - `GRADIENT_CLIP`: Exclusive optional executor for gradient clipping.
        - **Context**: same as `BEFORE_GRADIENT_CLIP`
        - **Return**: `None`
        - **Default**: no clipping

    - `BEFORE_OPTIMIZER_STEP`: Broadcast after gradient preparation and
      clipping, immediately before the optimizer operation.
        - **Context**: same as `BEFORE_GRADIENT_CLIP`
        - **Return**: ignored

    - `OPTIMIZER_STEP`: Exclusive executor for the optimizer operation.
        - **Context**: same as `BEFORE_OPTIMIZER_STEP`
        - **Return**: `None`
        - **Default**: `optimizer.step()`

    - `AFTER_OPTIMIZER_STEP`: Broadcast after the optimizer operation succeeds
      and Battery increments `optimizer_step_idx`. It is never emitted for
      intermediate accumulation batches or failed optimizer operations.
        - **Context**: same as `BEFORE_OPTIMIZER_STEP`, with the updated
          `optimizer_step_idx`
        - **Return**: ignored

    Model provider example:

    ```python
    @charge(Event.CONFIGURE_TRAIN_STEP)
    def configure_step(self, context: EventContext) -> OptimizationStep:
        return OptimizationStep()
    ```

    Callback execution-context example:

    ```python
    @charge(Event.STEP_EXECUTION_CONTEXT)
    def execution_context(self, context: EventContext):
        return torch.autocast(context["device"].type, dtype=torch.bfloat16)
    ```

    Exclusive executor example:

    ```python
    @charge(Event.BACKWARD)
    def backward(self, context: EventContext) -> None:
        context["backward_loss"].backward()
    ```

    ## Training Events

    - `BEFORE_TRAIN`: Called before training starts.
        - **Context**: `optimizer`

    - `AFTER_TRAIN`: Called after training completes.
        - **Context**: `optimizer`, `epoch`, `train_metrics`,
          `val_metrics` (if validation ran), `history_train_loss`,
          `history_val_loss`, `history_train_metrics`, `history_val_metrics`

    - `BEFORE_TRAIN_EPOCH`: Called before each training epoch.
        - **Context**: `optimizer`, `epoch`

    - `AFTER_TRAIN_EPOCH`: Called after each training epoch.
        - **Context**: `optimizer`, `epoch`, `train_metrics`,
          `history_train_loss`, `history_val_loss`, `history_train_metrics`,
          `history_val_metrics`

    - `BEFORE_TRAIN_STEP`: Called before each training batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `TRAIN_STEP`: Called for each training batch. Returns `StepOutput`, or a
      scalar loss when automatic Battery metrics are not configured.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `AFTER_TRAIN_STEP`: Called after each training batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`, `loss`,
          `train_loss`, `train_metrics`

    ## Validation Events

    - `BEFORE_VALIDATION`: Called before validation starts.
        - **Context**: `optimizer`, `epoch`, `train_metrics`,
          `history_train_loss`, `history_val_loss`, `history_train_metrics`,
          `history_val_metrics`

    - `AFTER_VALIDATION`: Called after validation completes.
        - **Context**: `optimizer`, `epoch`, `train_metrics`, `val_metrics`,
          `history_train_loss`, `history_val_loss`, `history_train_metrics`,
          `history_val_metrics`

    - `BEFORE_VALIDATION_EPOCH`: Called before each validation epoch.
        - **Context**: `epoch`

    - `AFTER_VALIDATION_EPOCH`: Called after each validation epoch.
        - **Context**: `epoch`, `val_metrics`

    - `BEFORE_VALIDATION_STEP`: Called before each validation batch.
        - **Context**: `batch`, `batch_idx`, `epoch`

    - `VALIDATION_STEP`: Called for each validation batch. Returns `StepOutput`,
      or a scalar loss when automatic Battery metrics are not configured.
        - **Context**: `batch`, `batch_idx`, `epoch`

    - `AFTER_VALIDATION_STEP`: Called after each validation batch.
        - **Context**: `batch`, `batch_idx`, `epoch`, `loss`, `val_loss`,
          `val_metrics`

    ## Test Events

    - `BEFORE_TEST`: Called before testing starts.
        - **Context**: `optimizer`

    - `AFTER_TEST`: Called after testing completes.
        - **Context**: `optimizer`, `loss`, `test_loss`, `test_metrics`

    - `BEFORE_TEST_EPOCH`: Called before test epoch.
        - **Context**: `optimizer`, `epoch`

    - `AFTER_TEST_EPOCH`: Called after test epoch.
        - **Context**: `optimizer`, `epoch`, `loss`, `test_loss`, `test_metrics`

    - `BEFORE_TEST_STEP`: Called before each test batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `TEST_STEP`: Called for each test batch. Returns `StepOutput`, or a scalar
      loss when automatic Battery metrics are not configured.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `AFTER_TEST_STEP`: Called after each test batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`, `loss`,
          `test_loss`, `test_metrics`

    ## Prediction Events

    - `BEFORE_PREDICT`: Called before prediction starts.
        - **Context**: `optimizer`

    - `AFTER_PREDICT`: Called after prediction completes.
        - **Context**: `optimizer`, `predictions`

    - `BEFORE_PREDICT_EPOCH`: Called before prediction epoch.
        - **Context**: `optimizer`, `epoch`

    - `AFTER_PREDICT_EPOCH`: Called after prediction epoch.
        - **Context**: `optimizer`, `epoch`, `predictions`

    - `BEFORE_PREDICT_STEP`: Called before each prediction batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `PREDICT_STEP`: Called for each prediction batch (must return predictions).
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `AFTER_PREDICT_STEP`: Called after each prediction batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`, `predictions`
    """

    # DataPack lifecycle events
    PREPARE_DATA = "prepare_data"
    SETUP_DATA = "setup_data"
    CONFIGURE_DATALOADER = "configure_dataloader"
    TEARDOWN_DATA = "teardown_data"

    # Optimization extension events
    SETUP = "setup"
    STEP_EXECUTION_CONTEXT = "step_execution_context"
    CONFIGURE_TRAIN_STEP = "configure_train_step"
    BEFORE_BACKWARD = "before_backward"
    BACKWARD = "backward"
    AFTER_BACKWARD = "after_backward"
    BEFORE_GRADIENT_CLIP = "before_gradient_clip"
    GRADIENT_CLIP = "gradient_clip"
    BEFORE_OPTIMIZER_STEP = "before_optimizer_step"
    OPTIMIZER_STEP = "optimizer_step"
    AFTER_OPTIMIZER_STEP = "after_optimizer_step"

    # Existing workflow lifecycle events
    BEFORE_TRAIN = "before_train"
    AFTER_TRAIN = "after_train"
    BEFORE_TRAIN_EPOCH = "before_train_epoch"
    AFTER_TRAIN_EPOCH = "after_train_epoch"
    BEFORE_TRAIN_STEP = "before_train_step"
    TRAIN_STEP = "train_step"
    AFTER_TRAIN_STEP = "after_train_step"

    # Validation lifecycle events
    BEFORE_VALIDATION = "before_validation"
    AFTER_VALIDATION = "after_validation"
    BEFORE_VALIDATION_EPOCH = "before_validation_epoch"
    AFTER_VALIDATION_EPOCH = "after_validation_epoch"
    BEFORE_VALIDATION_STEP = "before_validation_step"
    VALIDATION_STEP = "validation_step"
    AFTER_VALIDATION_STEP = "after_validation_step"

    # Test lifecycle events
    BEFORE_TEST = "before_test"
    AFTER_TEST = "after_test"
    BEFORE_TEST_EPOCH = "before_test_epoch"
    AFTER_TEST_EPOCH = "after_test_epoch"
    BEFORE_TEST_STEP = "before_test_step"
    TEST_STEP = "test_step"
    AFTER_TEST_STEP = "after_test_step"

    # Prediction lifecycle events
    BEFORE_PREDICT = "before_predict"
    AFTER_PREDICT = "after_predict"
    BEFORE_PREDICT_EPOCH = "before_predict_epoch"
    AFTER_PREDICT_EPOCH = "after_predict_epoch"
    BEFORE_PREDICT_STEP = "before_predict_step"
    PREDICT_STEP = "predict_step"
    AFTER_PREDICT_STEP = "after_predict_step"
