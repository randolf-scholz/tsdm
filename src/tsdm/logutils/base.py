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

Examples
--------
We want to log the left-inverse residual of the linodenet.
Sometimes, this is also used as a regularization term.
therefore the logging function either has to:

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
    "Logger",
    "BaseLogger",
    "StandardLogger",
    "DefaultLogger",
]

import pickle
import shutil
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import KW_ONLY, dataclass, field
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import pandas as pd
import torch
import yaml
from pandas import DataFrame, Index, MultiIndex
from torch import Tensor
from torch.nn import Module as TorchModule
from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import _LRScheduler as TorchLRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tsdm.encoders import BaseEncoder
from tsdm.logutils import (
    compute_metrics,
    log_kernel,
    log_metrics,
    log_model,
    log_optimizer,
    log_scalars,
)
from tsdm.logutils._callbacks import (
    Callback,
    CheckpointCallback,
    KernelCallback,
    MetricsCallback,
    ModelCallback,
    OptimizerCallback,
    ScalarsCallback,
    TableCallback,
)
from tsdm.metrics import Loss
from tsdm.types.aliases import PathLike
from tsdm.utils.strings import repr_mapping


class ResultTuple(NamedTuple):
    r"""Tuple of targets and predictions."""

    targets: Tensor
    predics: Tensor


@runtime_checkable
class Logger(Protocol):
    """Generic Logger Protocol."""

    @property
    def callbacks(self) -> Mapping[str, list[Callback]]:
        """Callbacks to be called at the end of a batch/epoch."""

    def run_callbacks(self, i: int, /, key: str, **kwargs: Any) -> None:
        """Call the logger."""


class BaseLogger:
    """Base class for loggers."""

    callbacks: dict[str, list[Callback]] = defaultdict(list)
    """Callbacks to be called at the end of a batch/epoch."""
    state_dict: dict[str, Any] = {}
    """State dictionary of the logger."""

    # def add_callback(self, key: str, callback: Callback | type[Callback], /, *args: Any, **kwargs: Any) -> None:
    #     """Add a callback to the logger."""
    #     if isinstance(callback, Callback):
    #         assert args == () and kwargs == {}
    #     else:
    #         callback = Callback(*args, **kwargs)
    #     self.callbacks[key].append(callback)

    @property
    def required_kwargs(self, key: str, /) -> list[set[str]]:
        return [callback.required_kwargs for callback in self.callbacks[key]]

    def run_callbacks(self, i: int, key: str, /, **kwargs: Any) -> None:
        """Call the logger."""
        required_kwargs = self.required_kwargs(key)

        # safety checks
        combined_kwargs = set.union(*required_kwargs)
        if missing_kwargs := (combined_kwargs - set(kwargs)):
            raise TypeError(f"Missing required kwargs: {missing_kwargs}")
        if unexpected_kwargs := (set(kwargs) - combined_kwargs):
            raise RuntimeWarning(f"Unexpected kwargs: {unexpected_kwargs}")

        # call callbacks
        for num, callback in enumerate(self.callbacks[key]):
            kwds = required_kwargs[num]
            callback(i, **{kwd: kwargs[kwd] for kwd in kwds})

    def __repr__(self) -> str:
        return repr_mapping(self.callbacks, wrapped=self)


class DefaultLogger(BaseLogger):
    r"""Logger for training and validation."""

    log_dir: Path
    """Path to the logging directory."""
    results_dir: Path
    """Path to the results directory."""
    checkpoint_dir: Path
    """Path to the checkpoint directory."""

    writer: SummaryWriter
    """Tensorboard writer."""
    callbacks: dict[str, list[Callback]] = defaultdict(list)
    """Callbacks to be called at the end of a batch/epoch."""
    frequency: dict[str, list[int]] = defaultdict(list)
    """How often to call the callbacks."""

    model: TorchModule
    optimizer: TorchOptimizer
    lr_scheduler: TorchLRScheduler

    def __init__(
        self,
        log_dir: PathLike,
        metrics: Optional[Mapping[str, Loss]] = None,
        checkpointable_objects: Optional[Mapping[str, Any]] = None,
        checkpoint_frequency: int = 1,
    ) -> None:
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = Path(self.writer.log_dir)

        # # add default batch callbacks
        # if self.optimizer is not None:
        #     self.add_callback(log_optimizer_state)
        # if self.metrics is not None:
        #     self.add_callback(log_metrics)
        #
        # # add default epoch callbacks
        # if self.model is not None:
        #     self.add_callback(log_model_state)
        # if self.optimizer is not None:
        #     self.add_callback(log_optimizer_state)
        # if self.lr_scheduler is not None:
        #     self.add_callback(log_lr_scheduler_state)
        # # if self.model is not None and hasattr(self.model, "kernel"):
        # #     self.add_callback(log_kernel_information)
        # if self.metrics is not None:
        #     self.add_callback(log_all_metrics)
        if checkpointable_objects is not None:
            self.add_callback(
                CheckpointCallback(
                    **checkpointable_objects, frequency=checkpoint_frequency
                )
            )
        #
        # # add results callbacks
        # self.add_callback(log_table, on="results")

    @torch.no_grad()
    def get_all_predictions(self, dataloader: DataLoader) -> ResultTuple:
        r"""Get all predictions for a dataloader."""
        # TODO: generic version
        targets = []
        predics = []

        for batch in dataloader:
            result = self.predict_fn(self.model, batch)
            targets.append(result[0])
            predics.append(result[1])

        return ResultTuple(
            # batch the list of batches by converting to list of single elements
            targets=torch.nn.utils.rnn.pad_sequence(
                chain.from_iterable(targets), batch_first=True, padding_value=torch.nan  # type: ignore[arg-type]
            ).squeeze(),
            predics=torch.nn.utils.rnn.pad_sequence(
                chain.from_iterable(predics), batch_first=True, padding_value=torch.nan  # type: ignore[arg-type]
            ).squeeze(),
        )

    def log_all_metrics(self, i: int) -> None:
        r"""Log all metrics for all dataloaders."""
        if i not in self.history.index:
            empty_row = DataFrame(
                index=[i], columns=self.history.columns, dtype="Float32"
            )
            self.history = pd.concat([self.history, empty_row], sort=True)

        if not self._warned_tuple:
            self._warned_tuple = True
            warnings.warn(
                "We recommend using a NamedTuple or a dictionary as the return type."
                " Ensure that 'targets'/'predics' is the first/second output.",
                UserWarning,
                stacklevel=2,
            )

        # Schema: identifier:category/key, e.g. `metrics:MSE/train`
        for key, dataloader in self.dataloaders.items():
            result = self.get_all_predictions(dataloader)
            scalars = compute_metrics(
                self.metrics, targets=result.targets, predics=result.predics
            )

            for metric, value in scalars.items():
                self.history.loc[i, (key, metric)] = value.cpu().item()

            log_scalars(i, writer=self.writer, scalars=scalars, key=key)

    def log_results(self, i: int, /) -> None:
        r"""Store history dataframe to file (default format: parquet)."""
        assert self.results_dir is not None
        path = self.results_dir / f"history-{i}.parquet"
        self.history.to_parquet(path)

    def log_hparams(self, i: int, /) -> None:
        r"""Log hyperparameters."""
        # Find the best epoch on the smoothed validation curve
        best_epochs = self.history.rolling(5, center=True).mean().idxmin()

        scores = {
            split: {
                metric: float(self.history.loc[idx, (split, metric)])
                for metric, idx in best_epochs["valid"].items()
            }
            for split in ("train", "valid", "test")
        }

        if self.results_dir is not None:
            with open(self.results_dir / f"{i}.yaml", "w", encoding="utf8") as file:
                file.write(yaml.dump(scores))

        # add postfix
        test_scores = {f"metrics:hparam/{k}": v for k, v in scores["test"].items()}

        self.writer.add_hparams(
            hparam_dict=self.hparam_dict, metric_dict=test_scores, run_name="hparam"
        )
        print(f"{test_scores=} achieved by {self.hparam_dict=}")

        # FIXME: https://github.com/pytorch/pytorch/issues/32651
        # os.path.dirname(os.path.realpath(__file__)) + os.sep + log_path
        for files in (self.logging_dir / "hparam").iterdir():
            shutil.move(files, self.logging_dir)
        (self.logging_dir / "hparam").rmdir()


@dataclass
class StandardLogger:
    r"""Logger for training and validation."""

    writer: SummaryWriter
    model: TorchModule

    _: KW_ONLY

    checkpoint_dir: Path
    dataloaders: Mapping[str, DataLoader]
    metrics: Mapping[str, Loss]
    predict_fn: Union[
        Callable[[TorchModule, tuple], tuple],
        Callable[[TorchModule, tuple], ResultTuple],
        # Callable[[TorchModule, tuple], ResultDict],
    ]
    results_dir: Optional[Path] = None

    encoder: BaseEncoder = NotImplemented
    hparam_dict: dict[str, Any] = NotImplemented
    optimizer: TorchOptimizer = NotImplemented
    lr_scheduler: TorchLRScheduler = NotImplemented

    # extra fields
    callbacks: dict[Literal["batch", "epoch"], list[Callback]] = field(
        init=False, default_factory=lambda: {"batch": [], "epoch": []}
    )
    history: DataFrame = field(init=False)
    logging_dir: Path = field(init=False)

    _warned_tuple: bool = False
    # TODO: add callbacks for batch, epoch, and run.
    # TODO: add multiple kernels logging.
    # idea: pass dict {key: Tensor}

    def __post_init__(self) -> None:
        r"""Initialize logger."""
        self.logging_dir = Path(self.writer.log_dir)
        columns = MultiIndex.from_product([self.dataloaders, self.metrics])
        index = Index([], name="epoch", dtype=int)
        self.history = DataFrame(index=index, columns=columns, dtype="Float32")

        if self.results_dir is not None:
            self.results_dir = Path(self.results_dir)

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        return f"{self.__class__.__name__}"

    # @torch.no_grad()
    # def get_all_predictions(self, dataloader: DataLoader) -> ResultTuple:
    #     r"""Get all predictions for a dataloader."""
    #     predics = []
    #     targets = []
    #
    #     # determine the type of the output of the predict_fn
    #     iloader = iter(dataloader)
    #     batch = next(iloader)
    #     result = self.predict_fn(self.model, batch)
    #
    #     if isinstance(result, dict):
    #         targets.append(result["targets"])
    #         predics.append(result["predics"])
    #         for batch in iloader:
    #             result = self.predict_fn(self.model, batch)
    #             targets.append(result["targets"])
    #             predics.append(result["predics"])
    #     elif isinstance(result, ResultTuple):
    #         targets.append(result.targets)
    #         predics.append(result.predics)
    #         for batch in iloader:
    #             result = self.predict_fn(self.model, batch)
    #             targets.append(result.targets)
    #             predics.append(result.predics)
    #     elif isinstance(result, tuple):
    #
    #         targets.append(result[0])
    #         predics.append(result[1])
    #         for batch in iloader:
    #             result = self.predict_fn(self.model, batch)
    #             targets.append(result[0]) e⃗
    #             predics.append(result[1])
    #
    #     masks = [~torch.isnan(p) for p in predics]
    #     return ResultTuple(
    #         targets=torch.cat([t[m] for t, m in zip(targets, masks)]).squeeze(),
    #         predics=torch.cat([p[m] for p, m in zip(predics, masks)]).squeeze(),
    #     )

    @torch.no_grad()
    def get_all_predictions(self, dataloader: DataLoader) -> ResultTuple:
        r"""Get all predictions for a dataloader."""
        # TODO: generic version
        targets = []
        predics = []

        for batch in dataloader:
            result = self.predict_fn(self.model, batch)
            targets.append(result[0])
            predics.append(result[1])

        return ResultTuple(
            # batch the list of batches by converting to list of single elements
            targets=torch.nn.utils.rnn.pad_sequence(
                chain.from_iterable(targets), batch_first=True, padding_value=torch.nan  # type: ignore[arg-type]
            ).squeeze(),
            predics=torch.nn.utils.rnn.pad_sequence(
                chain.from_iterable(predics), batch_first=True, padding_value=torch.nan  # type: ignore[arg-type]
            ).squeeze(),
        )

        # masks = [~torch.isnan(p) for p in predics]

        # return ResultTuple(
        #     targets=torch.cat([t[m] for t, m in zip(targets, masks)]).squeeze(),
        #     predics=torch.cat([p[m] for p, m in zip(predics, masks)]).squeeze(),
        # )

    def make_checkpoint(self, i: int, /) -> None:
        r"""Make checkpoint."""
        path = self.checkpoint_dir / f"{i}"
        path.mkdir(parents=True, exist_ok=True)

        # serialize the model
        if isinstance(self.model, torch.jit.ScriptModule):
            torch.jit.save(self.model, path / "model")
        else:
            torch.save(self.model, path / "model")

        # serialize the optimizer
        if self.optimizer is not NotImplemented:
            torch.save(self.optimizer, path / "optimizer")

        # serialize the lr scheduler
        if self.lr_scheduler is not NotImplemented:
            torch.save(self.lr_scheduler, path / "lr_scheduler")

        # serialize the hyperparameters
        if self.hparam_dict is not NotImplemented:
            with open(path / "hyperparameters.yaml", "w", encoding="utf8") as file:
                yaml.safe_dump(self.hparam_dict, file)

        # serialize the hyperparameters
        if self.encoder is not NotImplemented:
            with open(path / "encoder.pickle", "wb") as file:
                pickle.dump(self.encoder, file)

    def log_metrics(
        self,
        i: int,
        /,
        *,
        targets: Tensor,
        predics: Tensor,
        key: str = "",
        prefix: str = "",
        postfix: str = "",
    ) -> None:
        r"""Log metrics."""
        log_metrics(
            i,
            writer=self.writer,
            metrics=self.metrics,
            targets=targets,
            predics=predics,
            key=key,
            prefix=prefix,
            postfix=postfix,
        )

    def log_batch_end(
        self,
        i: int,
        *,
        targets: Optional[Tensor] = None,
        predics: Optional[Tensor] = None,
        loss: Optional[Tensor] = None,
    ) -> None:
        r"""Log batch end."""
        if targets is not None and predics is not None:
            self.log_metrics(i, targets=targets, predics=predics, key="batch")
        self.log_optimizer_state(i, loss=loss, log_scalars=True, log_histograms=False)

    def log_epoch_end(
        self,
        i: int,
        *,
        # individual switches
        log_kernel: bool | int = True,
        log_model: bool | int = True,
        log_optimizer: bool | int = False,
        make_checkpoint: bool | int = 10,
    ) -> None:
        r"""Log epoch end."""
        self.log_all_metrics(i)

        if log_kernel and i % log_kernel == 0:
            self.log_kernel_information(i)

        if log_model and i % log_model == 0:
            self.log_model_state(i, log_scalars=False, log_histograms=True)

        if log_optimizer and i % log_optimizer == 0:
            self.log_optimizer_state(i, log_scalars=False, log_histograms=True)

        if make_checkpoint and i % make_checkpoint == 0:
            self.make_checkpoint(i)

    def log_all_metrics(self, i: int) -> None:
        r"""Log all metrics for all dataloaders."""
        if i not in self.history.index:
            empty_row = DataFrame(
                index=[i], columns=self.history.columns, dtype="Float32"
            )
            self.history = pd.concat([self.history, empty_row], sort=True)

        if not self._warned_tuple:
            self._warned_tuple = True
            warnings.warn(
                "We recommend using a NamedTuple or a dictionary as the return type."
                " Ensure that 'targets'/'predics' is the first/second output.",
                UserWarning,
                stacklevel=2,
            )

        # Schema: identifier:category/key, e.g. `metrics:MSE/train`
        for key, dataloader in self.dataloaders.items():
            result = self.get_all_predictions(dataloader)
            scalars = compute_metrics(
                self.metrics, targets=result.targets, predics=result.predics
            )

            for metric, value in scalars.items():
                self.history.loc[i, (key, metric)] = value.cpu().item()

            log_scalars(
                i,
                writer=self.writer,
                scalars=scalars,
                key=key,
            )

    def log_hparams(self, i: int, /) -> None:
        r"""Log hyperparameters."""
        # Find the best epoch on the smoothed validation curve
        best_epochs = self.history.rolling(5, center=True).mean().idxmin()

        scores = {
            split: {
                metric: float(self.history.loc[idx, (split, metric)])
                for metric, idx in best_epochs["valid"].items()
            }
            for split in ("train", "valid", "test")
        }

        if self.results_dir is not None:
            with open(self.results_dir / f"{i}.yaml", "w", encoding="utf8") as file:
                file.write(yaml.dump(scores))

        # add postfix
        test_scores = {f"metrics:hparam/{k}": v for k, v in scores["test"].items()}

        self.writer.add_hparams(
            hparam_dict=self.hparam_dict, metric_dict=test_scores, run_name="hparam"
        )
        print(f"{test_scores=} achieved by {self.hparam_dict=}")

        # FIXME: https://github.com/pytorch/pytorch/issues/32651
        for files in (self.logging_dir / "hparam").iterdir():
            shutil.move(files, self.logging_dir)
        (self.logging_dir / "hparam").rmdir()

    def log_history(self, i: int, /) -> None:
        r"""Store history dataframe to file (default format: parquet)."""
        assert self.results_dir is not None

        path = self.results_dir / f"history-{i}.parquet"
        self.history.to_parquet(path)

    def log_optimizer_state(self, i: int, /, **kwds: Any) -> None:
        r"""Log optimizer state."""
        assert self.optimizer is not NotImplemented
        log_optimizer(i, writer=self.writer, optimizer=self.optimizer, **kwds)

    def log_model_state(self, i: int, /, **kwds: Any) -> None:
        r"""Log model state."""
        assert self.model is not NotImplemented
        log_model(i, writer=self.writer, model=self.model, **kwds)

    def log_kernel_information(self, i: int, /, **kwds: Any) -> None:
        r"""Log kernel information."""
        assert self.model is not NotImplemented
        assert hasattr(self.model, "kernel") and isinstance(self.model.kernel, Tensor)
        log_kernel(i, writer=self.writer, kernel=self.model.kernel, **kwds)
