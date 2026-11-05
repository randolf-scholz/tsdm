r"""Organize logging callbacks by training event.

`BaseLogger` groups callbacks under string keys. Its `log_batch_end`,
`log_epoch_end`, and `log_results` methods dispatch the `batch`, `epoch`, and
`result` groups, respectively. Each callback receives the current `step` and
keyword arguments from the supplied state, and applies its own frequency check.

Callbacks can hold mutable objects such as models and optimizers. Pass existing
losses, predictions, targets, or other computed results as keyword arguments to
reuse them during logging. `WrapCallback` adapts a callable to this interface.

Example:
    ```python
    from tsdm.logutils import BaseLogger, WrapCallback


    def log_loss(step: int, /, *, loss: float) -> None:
        print(f"Step {step}: loss={loss}")


    logger = BaseLogger()
    logger.add_callback("batch", WrapCallback(log_loss))
    logger.log_batch_end(1, loss=0.5)
    ```
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
from typing import Any, ClassVar, Optional, Protocol

from pandas import DataFrame, MultiIndex
from torch import Tensor
from torch.nn import Module as TorchModule
from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tsdm.constants import UNDEFINED
from tsdm.pprint import pprint_mapping, pprint_repr
from tsdm.types import FilePath
from tsdm.types.nested import JSON

from .callbacks import (
    Callback,
    CallbackList,
    CallbackSequence,
    CheckpointCallback,
    EvaluationCallback,
    HParamCallback,
    KernelCallback,
    LRSchedulerCallback,
    MetricsCallback,
    ModelCallback,
    OptimizerCallback,
)
from .utils import Loss


class Logger[CallbackT: CallbackSequence](Protocol):  # +CallbackT
    r"""Protocol for loggers that expose callback sequences by event key."""

    @property
    @abstractmethod
    def callbacks(self) -> Mapping[str, CallbackT]:
        r"""Callback sequences indexed by event key."""
        ...


@pprint_mapping
@dataclass(slots=True)
class BaseLogger(Mapping[str, CallbackList]):
    r"""Base class for loggers."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Diagnostic logger for this class."""

    callbacks: dict[str, CallbackList] = field(
        default_factory=lambda: defaultdict(CallbackList)
    )
    r"""Callback lists indexed by event key, such as `batch`, `epoch`, or `result`."""

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self) -> Iterator[str]:
        return iter(self.callbacks)

    def __getitem__(self, key: str, /) -> CallbackList:
        return self.callbacks[key]

    def add_callback(self, key: str, callback: Callback) -> None:
        r"""Append a callback to the group identified by `key`."""
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
@dataclass(slots=True)
class DefaultLogger(BaseLogger):
    r"""Register batch, epoch, and result callbacks for the supplied objects."""

    _: KW_ONLY

    log_dir: FilePath
    r"""Path to the logging directory."""
    results_dir: FilePath
    r"""Path to the results directory."""
    checkpoint_dir: FilePath
    r"""Path to the checkpoint directory."""

    # Optional Arguments
    checkpoint_frequency: int = 1
    r"""Positive number of epoch steps between checkpoints."""

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
    metrics: Optional[dict[str, Loss]] = None
    r"""Metrics used for evaluation."""
    model: Optional[TorchModule] = None
    r"""Model used for training."""
    optimizer: Optional[TorchOptimizer] = None
    r"""Optimizer used for training."""
    predict_fn: Optional[Callable[..., tuple[Tensor, Tensor]]] = None
    r"""Function called as `predict_fn(model, batch)` to return targets and predictions."""
    checkpointable_objects: Optional[dict[str, Any]] = None
    r"""Objects to be checkpointed."""
    # derived from other arguments
    writer: SummaryWriter = UNDEFINED
    r"""TensorBoard writer; created from `log_dir` when not supplied."""

    def __post_init__(self) -> None:
        r"""Initialize the writer, evaluation history, and default callback groups."""
        self.writer = (
            SummaryWriter(log_dir=self.log_dir)
            if self.writer is UNDEFINED
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
        r"""Yield the default batch callbacks."""
        if self.metrics is not None:
            yield MetricsCallback(self.metrics, writer=self.writer, key="batch")
        if self.optimizer is not None:
            yield OptimizerCallback(
                self.optimizer, writer=self.writer, log_histograms=False
            )

    def make_default_result_callbacks(self) -> Iterator[Callback]:
        r"""Yield the default result callbacks."""
        if self.history is not None and self.hparam_dict is not None:
            yield HParamCallback(
                self.hparam_dict, history=self.history, writer=self.writer
            )

    def make_default_epoch_callbacks(self) -> Iterator[Callback]:
        r"""Yield the default epoch callbacks."""
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
        if self.model is not None:
            yield ModelCallback(
                self.model,
                writer=self.writer,
                log_norms=True,
                log_histograms=True,
            )
