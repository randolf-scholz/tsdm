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
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pandas import DataFrame, MultiIndex
from torch import Tensor
from torch.nn import Module as TorchModule
from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import _LRScheduler as TorchLRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import (
    Any,
    ClassVar,
    Optional,
    Protocol,
    runtime_checkable,
)

from tsdm.logutils.callbacks import (
    Callback,
    CheckpointCallback,
    EvaluationCallback,
    HParamCallback,
    KernelCallback,
    LRSchedulerCallback,
    MetricsCallback,
    OptimizerCallback,
)
from tsdm.metrics import Metric
from tsdm.types.aliases import JSON, PathLike
from tsdm.utils.funcutils import get_mandatory_kwargs
from tsdm.utils.pprint import repr_mapping


@runtime_checkable
class Logger(Protocol):
    """Generic Logger Protocol."""

    @property
    def callbacks(self) -> Mapping[str, Sequence[Callback]]:
        """Callbacks to be called at the end of a batch/epoch."""
        ...

    def callback(self, key: str, i: int, /, **kwargs: Any) -> None:
        """Call the logger."""
        ...


class BaseLogger(Logger):
    """Base class for loggers."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    """The debug logger."""

    _callbacks: dict[str, list[tuple[Callback[...], int, set[str]]]] = defaultdict(list)
    """Callbacks to be called at the end of a batch/epoch."""

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "LOGGER"):
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        return super().__new__(cls)

    @property
    def callbacks(self) -> dict[str, list[Callback]]:
        """Callbacks to be called at the end of a batch/epoch."""
        return {k: [cb for cb, _, _ in v] for k, v in self._callbacks.items()}

    def add_callback(
        self, key: str, callback: Callback, /, *, frequency: Optional[int] = None
    ) -> None:
        """Add a callback to the logger."""
        required_kwargs = set(
            callback.required_kwargs
            if hasattr(callback, "required_kwargs")
            else get_mandatory_kwargs(callback)
        )

        # set the frequency
        freq = getattr(callback, "frequency", 1) if frequency is None else frequency

        self._callbacks[key].append((callback, freq, required_kwargs))

    def combined_kwargs(self, key: str) -> set[str]:
        """Get the combined kwargs of all callbacks."""
        return set().union(*(kwargs for _, _, kwargs in self._callbacks[key]))

    def callback(self, key: str, i: int, /, **kwargs: Any) -> None:
        """Call the logger."""
        combined_kwargs = self.combined_kwargs(key)
        if missing_kwargs := combined_kwargs - set(kwargs):
            raise TypeError(f"Missing required kwargs: {missing_kwargs}")
        if unexpected_kwargs := set(kwargs) - combined_kwargs:
            raise RuntimeWarning(f"Unexpected kwargs: {unexpected_kwargs}")

        # call callbacks
        for cb, freq, callback_kwargs in self._callbacks[key]:
            if i % freq:
                self.LOGGER.debug("skipping callback %s", cb)
                continue
            self.LOGGER.debug("executing callback %s", cb)
            cb(i, **{k: kwargs[k] for k in callback_kwargs})

    def log_epoch_end(self, i: int, /, **kwargs: Any) -> None:
        """Log the end of an epoch."""
        self.callback("epoch", i, **kwargs)

    def log_batch_end(self, i: int, /, **kwargs: Any) -> None:
        """Log the end of a batch."""
        self.callback("batch", i, **kwargs)

    def log_results(self, i: int, /, **kwargs: Any) -> None:
        """Log the results of the experiment."""
        self.callback("results", i, **kwargs)

    def __repr__(self) -> str:
        return repr_mapping(self.callbacks, wrapped=self)


class DefaultLogger(BaseLogger):
    r"""Logger that adds pre-made batch/epoch logging."""

    log_dir: Path
    """Path to the logging directory."""
    results_dir: Path
    """Path to the results directory."""
    checkpoint_dir: Path
    """Path to the checkpoint directory."""
    checkpoint_frequency: int
    """Frequency of checkpoints."""

    config: Optional[JSON] = None
    """Configuration of the experiment."""
    dataloaders: Optional[Mapping[str, DataLoader]] = None
    """Dataloaders used for evaluation."""
    kernel: Optional[Tensor] = None
    """Kernel used for the model."""
    history: Optional[DataFrame] = None
    """History of the training process."""
    hparam_dict: Optional[dict[str, Any]] = None
    """Hyperparameters used for the experiment."""
    lr_scheduler: Optional[TorchLRScheduler] = None
    """Learning rate scheduler."""
    metrics: Optional[Mapping[str, Metric]] = None
    """Metrics used for evaluation."""
    model: Optional[TorchModule] = None
    """Model used for training."""
    optimizer: Optional[TorchOptimizer] = None
    """Optimizer used for training."""
    predict_fn: Optional[Callable[..., tuple[Tensor, Tensor]]] = None
    """Function used for prediction."""

    def __init__(
        self,
        log_dir: PathLike,
        checkpoint_dir: PathLike,
        results_dir: PathLike,
        *,
        checkpoint_frequency: int = 1,
        checkpointable_objects: Optional[Mapping[str, Any]] = None,
        config: Optional[JSON] = None,
        dataloaders: Optional[Mapping[str, DataLoader]] = None,
        hparam_dict: Optional[dict[str, Any]] = None,
        kernel: Optional[Tensor] = None,
        lr_scheduler: Optional[TorchLRScheduler] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
        model: Optional[TorchModule] = None,
        optimizer: Optional[TorchOptimizer] = None,
        predict_fn: Optional[Callable[..., tuple[Tensor, Tensor]]] = None,
    ) -> None:
        # convert to an absolute path
        self.log_dir = Path(log_dir).absolute()
        self.checkpoint_dir = Path(checkpoint_dir).absolute()
        self.results_dir = Path(results_dir).absolute()

        self.config = config
        self.dataloaders = dataloaders
        self.hparam_dict = hparam_dict
        self.kernel = kernel
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics
        self.model = model
        self.optimizer = optimizer
        self.predict_fn = predict_fn
        self.writer = SummaryWriter(log_dir=self.log_dir)

        checkpointable_objects = (
            {} if checkpointable_objects is None else checkpointable_objects
        )

        # region default batch callbacks -----------------------------------------------
        if self.metrics is not None:
            self.add_callback(
                "batch",
                MetricsCallback(
                    self.metrics,
                    self.writer,
                    key="batch",
                ),
            )
        if self.optimizer is not None:
            self.add_callback(
                "batch",
                OptimizerCallback(
                    self.optimizer,
                    self.writer,
                    log_histograms=False,
                ),
            )
        # endregion --------------------------------------------------------------------

        # region default epoch callbacks -----------------------------------------------
        if (
            self.metrics is not None
            and self.model is not None
            and self.dataloaders is not None
            and self.predict_fn is not None
        ):
            self.history = DataFrame(
                columns=MultiIndex.from_product([dataloaders, metrics])
            )
            self.add_callback(
                "epoch",
                EvaluationCallback(
                    model=self.model,
                    writer=self.writer,
                    metrics=self.metrics,
                    dataloaders=self.dataloaders,
                    history=self.history,
                    predict_fn=self.predict_fn,
                ),
            )

        # if self.model is not None:
        #     self.add_callback("epoch", ModelCallback(self.model, self.writer))

        # if self.optimizer is not None:
        #     self.add_callback("epoch", OptimizerCallback(self.optimizer, self.writer))

        if self.lr_scheduler is not None:
            self.add_callback(
                "epoch", LRSchedulerCallback(self.lr_scheduler, self.writer)
            )
        if self.kernel is not None:
            self.add_callback("epoch", KernelCallback(self.kernel, self.writer))

        self.add_callback(
            "epoch",
            CheckpointCallback(
                {
                    "config": self.config,
                    "model": self.model,
                    "optimizer": self.optimizer,
                    "lr_scheduler": self.lr_scheduler,
                }
                | dict(checkpointable_objects),
                path=self.checkpoint_dir,
                frequency=checkpoint_frequency,
            ),
        )
        # endregion --------------------------------------------------------------------

        # region default results callbacks ---------------------------------------------
        if self.history is not None and self.hparam_dict is not None:
            self.add_callback(
                "results",
                HParamCallback(
                    hparam_dict=self.hparam_dict,
                    history=self.history,
                    results_dir=self.results_dir,
                    writer=self.writer,
                ),
            )
        # endregion --------------------------------------------------------------------
