r"""Logging Utility Functions.

We allow the user to define callbacks for logging, which are called at the end of
batches/epochs.

A callback is a function that takes the following arguments:

- ``i``: The current iteration/epoch.
- ``writer``: The ``SummaryWriter`` instance.
- ``logged_object``: The object that is being logged.
    - what if we need to pass multiple objects?
- when used in callbacks, the
- Should a callback be self-contained?
- Does the callback need to reference the Logger it is being called from?

Idea: We want to so something like this:

.. code-block:: python

    logger = Logger(...)
    logger.add_callback(log_loss, on="batch")

How do we ensure logging is fast?

- loggers should not recompute things that have already been computed
    - loggers need access to existing results

Examples:
    We want to log the left-inverse residual of the linodenet.
    Sometimes, this is also used as a regularization term.
    Therefore, the logging function either has to:

    - compute R and log it
    - use the existing R and log it

    Typical things the loggers need access to include:

    - the current iteration
    - the current loss
    - the current model
    - the current optimizer
    - the current data
        - the current predictions
        - the current targets
    - the current metrics
"""

__all__ = [
    # ABCs & Protocols
    "Logger",
    "BaseLogger",
    #  Classes
    "DefaultLogger",
]

import logging
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import KW_ONLY, dataclass, field

from pandas import DataFrame, MultiIndex
from torch import Tensor
from torch.nn import Module as TorchModule
from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import (
    Any,
    ClassVar,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from tsdm.logutils.callbacks import (
    Callback,
    CallbackList,
    CallbackSequence,
    CheckpointCallback,
    EvaluationCallback,
    HParamCallback,
    KernelCallback,
    LRSchedulerCallback,
    MetricsCallback,
    OptimizerCallback,
)
from tsdm.metrics import Metric
from tsdm.types.aliases import JSON, FilePath
from tsdm.utils.decorators import pprint_mapping, pprint_repr

CBS_co = TypeVar("CBS_co", covariant=True, bound=CallbackSequence)


@runtime_checkable
class Logger(Protocol[CBS_co]):
    r"""Generic Logger Protocol."""

    @property
    @abstractmethod
    def callbacks(self) -> Mapping[str, CBS_co]:
        r"""Callbacks to be called at the end of a batch/epoch."""
        ...


@pprint_mapping
@dataclass
class BaseLogger(Mapping[str, CallbackList]):
    r"""Base class for loggers."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for the Encoder."""

    callbacks: dict[str, CallbackList] = field(
        default_factory=lambda: defaultdict(CallbackList)
    )
    r"""Callbacks to be called at the end of a batch/epoch."""

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self) -> Iterator[str]:
        return iter(self.callbacks)

    def __getitem__(self, key: str, /) -> CallbackList:
        return self.callbacks[key]

    def add_callback(self, key: str, callback: Callback) -> None:
        r"""Add a callback to the logger."""
        self.callbacks[key].append(callback)

    def log_epoch_end(self, step: int, /, **state_dict: Any) -> None:
        r"""Log the end of an epoch."""
        self["epoch"].callback(step, **state_dict)

    def log_batch_end(self, step: int, /, **state_dict: Any) -> None:
        r"""Log the end of a batch."""
        self["batch"].callback(step, **state_dict)

    def log_results(self, step: int, /, **state_dict: Any) -> None:
        r"""Log the results of the experiment."""
        self["result"].callback(step, **state_dict)


@pprint_repr
@dataclass(repr=False)
class DefaultLogger(BaseLogger):
    r"""Logger that adds pre-made batch/epoch logging."""

    _: KW_ONLY

    log_dir: FilePath
    r"""Path to the logging directory."""
    results_dir: FilePath
    r"""Path to the results directory."""
    checkpoint_dir: FilePath
    r"""Path to the checkpoint directory."""

    # Optional Arguments
    checkpoint_frequency: int = 1
    r"""Frequency of checkpoints."""

    config: Optional[JSON] = None
    r"""Configuration of the experiment."""
    dataloaders: Optional[dict[str, DataLoader]] = None
    r"""Dataloaders used for evaluation."""
    kernel: Optional[Tensor] = None
    r"""Kernel used for the model."""
    history: Optional[DataFrame] = None
    r"""History of the training process."""
    hparam_dict: Optional[dict[str, Any]] = None
    r"""Hyperparameters used for the experiment."""
    lr_scheduler: Optional[LRScheduler] = None
    r"""Learning rate scheduler."""
    metrics: Optional[dict[str, Metric]] = None
    r"""Metrics used for evaluation."""
    model: Optional[TorchModule] = None
    r"""Model used for training."""
    optimizer: Optional[TorchOptimizer] = None
    r"""Optimizer used for training."""
    predict_fn: Optional[Callable[..., tuple[Tensor, Tensor]]] = None
    r"""Function used for prediction."""
    checkpointable_objects: Optional[dict[str, Any]] = None
    r"""Objects to be checkpointed."""
    # derived from other arguments
    writer: SummaryWriter = NotImplemented
    r"""SummaryWriter used for logging."""

    def __post_init__(self) -> None:
        r"""Post-initialization steps."""
        self.writer = (
            SummaryWriter(log_dir=self.log_dir)
            if self.writer is NotImplemented
            else self.writer
        )

        # add default objects to checkpointable objects
        objects: dict[str, Any] = {}
        for key in ("config", "model", "optimizer", "lr_scheduler"):
            if (val := getattr(self, key)) is not None:
                objects[key] = val

        if self.checkpointable_objects is not None and objects:
            self.checkpointable_objects = objects | dict(self.checkpointable_objects)

        if (
            self.history is None
            and self.metrics is not None
            and self.dataloaders is not None
        ):
            self.history = DataFrame(
                columns=MultiIndex.from_product([self.dataloaders, self.metrics])
            )

        # set default callbacks
        self.callbacks["batch"].extend(self.make_default_batch_callbacks())
        self.callbacks["epoch"].extend(self.make_default_epoch_callbacks())
        self.callbacks["result"].extend(self.make_default_result_callbacks())

    def make_default_batch_callbacks(self) -> Iterator[Callback]:
        r"""Make the default batch callbacks."""
        if self.metrics is not None:
            yield MetricsCallback(self.metrics, writer=self.writer, key="batch")
        if self.optimizer is not None:
            yield OptimizerCallback(
                self.optimizer, writer=self.writer, log_histograms=False
            )

    def make_default_result_callbacks(self) -> Iterator[Callback]:
        r"""Make the default result callbacks."""
        if self.history is not None and self.hparam_dict is not None:
            yield HParamCallback(
                self.hparam_dict, history=self.history, writer=self.writer
            )

    def make_default_epoch_callbacks(self) -> Iterator[Callback]:
        r"""Make the default epoch callbacks."""
        if (
            self.metrics is not None
            and self.model is not None
            and self.dataloaders is not None
            and self.predict_fn is not None
            and self.history is not None
        ):
            yield EvaluationCallback(
                model=self.model,
                writer=self.writer,
                metrics=self.metrics,
                dataloaders=self.dataloaders,
                history=self.history,
                predict_fn=self.predict_fn,
            )

        if self.lr_scheduler is not None:
            yield LRSchedulerCallback(self.lr_scheduler, writer=self.writer)
        if self.kernel is not None:
            yield KernelCallback(self.kernel, writer=self.writer)
        if self.checkpointable_objects is not None:
            yield CheckpointCallback(
                self.checkpointable_objects,
                path=self.checkpoint_dir,
                frequency=self.checkpoint_frequency,
            )
