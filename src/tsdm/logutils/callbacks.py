r"""Callback utilities for logging.

A callback class generally should take a single positional argument to initialize.
This argument can be either:

1. A mutable object that changes by itself over time, e.g. a `DataFrame`, `nn.Module`, etc.
2. A `Callable[[int, **kwargs], T]` that takes a step index and returns a value to log.
"""

__all__ = [
    # ABCs & Protocols & Structural classes
    "BaseCallback",
    "Callback",
    "CallbackSequence",
    "CallbackList",
    # Callbacks
    "CheckpointCallback",
    "ConfigCallback",
    "EvaluationCallback",
    "HParamCallback",
    "KernelCallback",
    "LRSchedulerCallback",
    "MetricsCallback",
    "ModelCallback",
    "OptimizerCallback",
    "ScalarsCallback",
    "TableCallback",
    "WrapCallback",
    "PlotCallback",
    # Functions
    "is_callback",
]

import inspect
import logging
from abc import abstractmethod
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
from dataclasses import KW_ONLY, dataclass, field
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Concatenate,
    Literal,
    Optional,
    Protocol,
    Self,
    SupportsIndex,
    TypeIs,
    final,
    overload,
    runtime_checkable,
)

import torch
import yaml
from matplotlib.figure import Figure
from pandas import DataFrame, MultiIndex
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from tsdm.constants import EMPTY_MAP, UNDEFINED
from tsdm.logutils.logfuncs import (
    log_config,
    log_kernel,
    log_lr_scheduler,
    log_metrics,
    log_model,
    log_optimizer,
    log_plot,
    log_table,
    log_values,
)
from tsdm.logutils.utils import TargetsAndPredics, compute_metrics, save_checkpoint
from tsdm.metrics import Metric
from tsdm.optimizers import Optimizer
from tsdm.types.abc import MutSeq
from tsdm.types.aliases import JSON, FilePath
from tsdm.utils.decorators import pprint_repr, pprint_sequence
from tsdm.utils.funcutils import get_mandatory_kwargs


@runtime_checkable
class Callback(Protocol):
    r"""Protocol for callbacks.

    Callbacks should be initialized with mutable objects that are being logged.
    every time the callback is called, it should log the current state of the
    mutable object.
    """

    frequency: int = 1
    r"""The frequency at which the callback should be called."""

    @property
    @abstractmethod
    def required_kwargs(self) -> set[str]:
        r"""The required kwargs for the callback."""
        ...

    @abstractmethod
    def callback(self, step: int, /, **state_dict: Any) -> None:
        r"""Log something at time index i."""
        ...


@runtime_checkable
class CallbackSequence(MutSeq[Callback], Callback, Protocol):
    r"""Protocol for sequences of callbacks.

    Helper Protocol, in practice, this should be the intersection
    of `MutableSequence[Callback]` and `Callback`.
    """


def is_callback(obj: object, /) -> TypeIs[Callback]:
    r"""Check if the function is a callback."""
    if not callable(obj):
        return False

    try:
        sig = inspect.signature(obj)
    except Exception as exc:
        raise ValueError(f"Could not inspect signature of {obj=}") from exc

    params = list(sig.parameters.values())
    return (
        len(params) >= 1
        and params[0].kind is inspect.Parameter.POSITIONAL_ONLY
        and all(p.kind in {p.KEYWORD_ONLY, p.VAR_KEYWORD} for p in params[1:])
    )


@pprint_repr
@dataclass(repr=False)
class BaseCallback(Callback):
    r"""Base class for callbacks."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for the class."""

    _: KW_ONLY

    frequency: int = 1
    r"""The frequency at which the callback is executed."""

    @property
    def required_kwargs(self) -> set[str]:
        r"""The required kwargs for the callback."""
        return get_mandatory_kwargs(self.__call__)

    @abstractmethod
    def __call__(self, step: int, /, **state_dict: Any) -> None:
        r"""Implementation of the callback."""
        ...

    @final
    def callback(self, step: int, /, **state_dict: Any) -> None:
        r"""Executes the callback if `step` is a multiple of the frequency."""
        if step % self.frequency == 0:
            self.LOGGER.debug("Logging at step %d", step)
            self(step, **state_dict)


@pprint_sequence
class CallbackList(MutableSequence[Callback], BaseCallback):
    r"""Callback to log multiple callbacks."""

    callbacks: list[Callback]
    r"""The callbacks to log."""

    @property
    def required_kwargs(self) -> set[str]:
        r"""The required kwargs for the callback."""
        return set().union(*(callback.required_kwargs for callback in self.callbacks))

    def __init__(self, iterable: Iterable[Callback] = (), /) -> None:
        r"""Initialize the callback."""
        self.callbacks = list(iterable)

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self) -> Iterator[Callback]:
        return iter(self.callbacks)

    @overload
    def __getitem__(self, index: int, /) -> Callback: ...
    @overload
    def __getitem__(self, index: slice, /) -> Self: ...
    def __getitem__(self, index: int | slice, /) -> Callback | Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(index, SupportsIndex):
            return self.callbacks[index]
        return self.__class__(self.callbacks[index])

    @overload
    def __setitem__(self, index: int, value: Callback, /) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[Callback], /) -> None: ...
    def __setitem__(self, index: int | slice, value: Any, /) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.callbacks[index] = value

    @overload
    def __delitem__(self, index: int, /) -> None: ...
    @overload
    def __delitem__(self, index: slice, /) -> None: ...
    def __delitem__(self, index: int | slice, /) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        del self.callbacks[index]

    def __call__(self, step: int, /, **state_dict: Any) -> None:
        r"""Log something at the end of a batch/epoch."""
        for obj in self.callbacks:
            obj.callback(step, **state_dict)

    def insert(self, index: int, value: Callback) -> None:
        self.callbacks.insert(index, value)


@dataclass
class WrapCallback(BaseCallback):
    r"""Wraps callable as a callback."""

    func: Callable[Concatenate[int, ...], None]

    def __call__(self, step: int, /, **state_dict: Any) -> None:
        self.func(step, **state_dict)


@dataclass
class ConfigCallback(BaseCallback):
    r"""Callback to log the config to tensorboard."""

    config: JSON

    _: KW_ONLY

    writer: SummaryWriter | Path

    # Optional parameters
    fmt: Literal["json", "yaml", "toml"] = "yaml"
    name: str = "config"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, step: int, /, **_: Any) -> None:
        r"""Log the config to tensorboard."""
        log_config(
            step,
            self.writer,
            config=self.config,
            fmt=self.fmt,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class EvaluationCallback(BaseCallback):
    r"""Callback to log evaluation metrics to tensorboard."""

    model: nn.Module  # mutable

    _: KW_ONLY

    predict_fn: Callable[..., tuple[Tensor, Tensor]]
    dataloaders: Mapping[str, DataLoader]
    metrics: (
        Sequence[str | Metric | type[Metric]]
        | Mapping[str, str | Metric | type[Metric]]
    )
    writer: SummaryWriter

    # Optional parameters
    # index:Index[step], columns:MultiIndex[key, metric]
    history: DataFrame = UNDEFINED
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    # non-init fields
    # index:Index[step], columns:MultiIndex[key, metric]
    best_epoch: DataFrame = field(init=False)

    def __post_init__(self) -> None:
        r"""Initialize the callback."""
        if self.history is UNDEFINED:
            self.history = DataFrame(
                columns=MultiIndex.from_product([self.dataloaders, self.metrics])
            )
        self.best_epoch = self.history.copy()

        for i in self.history.index:
            self.compute_results(i)

        # make sure there is exactly one validation split.
        candidate = None
        for key in self.dataloaders:
            if key.lower() in {"val", "valid", "validation"}:
                if candidate is not None:
                    raise ValueError(
                        f"Found multiple validation splits: {candidate} and {key}!"
                    )
                candidate = key
        if candidate is None:
            raise ValueError("Could not find a validation split!")
        self._val_key = candidate

    def __call__(self, step: int, /, **_: Any) -> None:
        # update the history
        self.compute_results(step)
        self.compute_best_epoch(step)

        for key in self.dataloaders:
            log_values(
                step,
                self.writer,
                scalars=self.history.loc[step, key].to_dict(),
                key=key,
                name=self.name,
                prefix=self.prefix,
                postfix=self.postfix,
            )
            log_values(
                step,
                self.writer,
                scalars=self.best_epoch.loc[step, key].to_dict(),
                key=key,
                name="score",
                prefix=self.prefix,
                postfix=self.postfix,
            )

        path = Path(self.writer.log_dir) / f"history-{step}.parquet"
        self.history.to_parquet(path)

    def compute_results(self, step: int) -> None:
        r"""Return the results at the minimal validation loss."""
        for key, dataloader in self.dataloaders.items():
            result = self.get_all_predictions(dataloader)
            scalars = compute_metrics(
                self.metrics, targets=result.targets, predics=result.predics
            )
            for metric, value in scalars.items():
                self.history.loc[step, (key, metric)] = value.cpu().item()

    def compute_best_epoch(self, step: int) -> None:
        r"""Compute the best epoch for the given index."""
        # unoptimized...
        mask = self.history.index <= step
        best_epochs = (
            self.history
            .loc[mask, self._val_key]
            .rolling(5, min_periods=1)
            .mean()
            .idxmin()
        )
        for key in self.dataloaders:
            for metric in self.metrics:
                best_value = self.history.loc[best_epochs[metric], (key, metric)]
                self.best_epoch.loc[step, (key, metric)] = best_value

    @torch.no_grad()
    def get_all_predictions(self, dataloader: DataLoader) -> TargetsAndPredics:
        r"""Return the targets and predictions for the given dataloader."""
        targets_list: list[Tensor] = []
        predics_list: list[Tensor] = []

        for batch in tqdm(dataloader, desc="Evaluation", leave=False):
            target, predic = self.predict_fn(self.model, batch)
            targets_list.append(target)
            predics_list.append(predic)

        targets = torch.nn.utils.rnn.pad_sequence(
            list(chain.from_iterable(targets_list)),
            batch_first=True,
            padding_value=torch.nan,
        ).squeeze()

        predics = torch.nn.utils.rnn.pad_sequence(
            list(chain.from_iterable(predics_list)),
            batch_first=True,
            padding_value=torch.nan,
        ).squeeze()

        return TargetsAndPredics(targets=targets, predics=predics)


@dataclass
class CheckpointCallback(BaseCallback):
    r"""Callback to save checkpoints."""

    objects: Mapping[str, object]  # mutable objects to snapshot

    _: KW_ONLY

    path: FilePath

    def __call__(self, step: int, /, **_: Any) -> None:
        save_checkpoint(step, self.path, objects=self.objects)


@dataclass
class HParamCallback(BaseCallback):
    r"""Callback to log hyperparameters to tensorboard."""

    hparam_dict: JSON

    _: KW_ONLY

    history: DataFrame
    r"""The history of the metrics, must have columns for train, valid, and test."""
    writer: SummaryWriter

    def __post_init__(self) -> None:
        r"""Validate the inputs."""
        if not isinstance(self.history, DataFrame):
            raise TypeError("histry must be a DataFrame!")
        if not isinstance(self.history.columns, MultiIndex):
            raise TypeError("history columns must be a MultiIndex!")

        splits = set(self.history.columns.levels[0])
        metrics = set(self.history.columns.levels[1])

        if splits != {"train", "valid", "test"}:
            raise ValueError("splits must be {'train', 'valid', 'test'}")

        if set(self.history.columns) != set(MultiIndex.from_product([splits, metrics])):
            raise ValueError(
                f"history columns must be a MultiIndex with {splits=} and {metrics=}"
            )

    def __call__(self, step: int, /, **_: Any) -> None:
        scores = self.scores
        metric_dict = {f"metrics:hparam/{k}": v for k, v in scores["test"].items()}

        run_name = Path(self.writer.log_dir).absolute()
        path = run_name / f"{step}.yaml"

        with path.open("w", encoding="utf8") as file:
            yaml.safe_dump(scores, file)

        self.writer.add_hparams(
            hparam_dict=self.hparam_dict,
            metric_dict=metric_dict,
            run_name=run_name,
        )

    @property
    def scores(self) -> dict[str, dict[str, float]]:
        r"""Return the current scores."""
        best_epochs = self.history.rolling(5, center=True).mean().idxmin()

        return {
            split: {
                str(metric): float(self.history.loc[idx, (split, metric)])
                for metric, idx in best_epochs["valid"].items()
            }
            for split in ("train", "valid", "test")
        }


@dataclass
class KernelCallback(BaseCallback):
    r"""Callback to log kernel information to tensorboard."""

    kernel: Tensor  # mutable

    _: KW_ONLY

    writer: SummaryWriter

    # Optional Parameters
    log_figures: bool = True
    log_scalars: bool = True

    # individual switches
    log_distances: bool = True
    log_heatmap: bool = True
    log_linalg: bool = True
    log_norms: bool = True
    log_scaled_norms: bool = True
    log_spectrum: bool = True

    name: str = "kernel"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, step: int, /, **_: Any) -> None:
        log_kernel(
            step,
            self.writer,
            kernel=self.kernel,
            log_figures=self.log_figures,
            log_scalars=self.log_scalars,
            log_distances=self.log_distances,
            log_heatmap=self.log_heatmap,
            log_linalg=self.log_linalg,
            log_norms=self.log_norms,
            log_scaled_norms=self.log_scaled_norms,
            log_spectrum=self.log_spectrum,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class LRSchedulerCallback(BaseCallback):
    r"""Callback to log learning rate information to tensorboard."""

    lr_scheduler: LRScheduler  # mutable

    _: KW_ONLY

    writer: SummaryWriter

    # Optional Parameters
    name: str = "lr_scheduler"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, step: int, /, **_: Any) -> None:
        log_lr_scheduler(
            step,
            self.writer,
            lr_scheduler=self.lr_scheduler,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class MetricsCallback(BaseCallback):
    r"""Callback to log multiple metrics to tensorboard."""

    metrics: Mapping[str, str | Metric | type[Metric]]

    _: KW_ONLY

    writer: SummaryWriter

    # Optional Parameters
    key: str = ""
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    def __call__(  # type: ignore[override]
        self, step: int, /, *, targets: Tensor, predictions: Tensor, **_: Any
    ) -> None:
        log_metrics(
            step,
            self.writer,
            metrics=self.metrics,
            targets=targets,
            predics=predictions,
            key=self.key,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class ModelCallback(BaseCallback):
    r"""Callback to log model information to tensorboard."""

    model: nn.Module  # mutable

    _: KW_ONLY

    writer: SummaryWriter

    # Optional Parameters
    log_histograms: bool = False
    log_norms: bool = True
    name: str = "model"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, step: int, /, **state_dict: Any) -> None:
        if "model" in state_dict:
            self.model = state_dict["model"]
        log_model(
            step,
            self.writer,
            model=self.model,
            log_histograms=self.log_histograms,
            log_norms=self.log_norms,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class OptimizerCallback(BaseCallback):
    r"""Callback to log optimizer information to tensorboard."""

    optimizer: Optimizer  # mutable

    _: KW_ONLY

    writer: SummaryWriter

    # Optional Parameters
    log_histograms: bool = False
    log_norms: bool = True
    name: str = "optimizer"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, step: int, /, **state_dict: Any) -> None:
        if "optimizer" in state_dict:
            self.optimizer = state_dict["optimizer"]
        log_optimizer(
            step,
            self.writer,
            optimizer=self.optimizer,
            log_histograms=self.log_histograms,
            log_norms=self.log_norms,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class ScalarsCallback(BaseCallback):
    r"""Callback to log multiple values to tensorboard."""

    scalars: dict[str, Tensor]

    _: KW_ONLY

    writer: SummaryWriter

    # Optional Parameters
    key: str = ""
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, step: int, /, **state_dict: Any) -> None:
        for key, value in state_dict.items():
            if key in self.scalars:
                self.scalars[key] = value
        log_values(
            step,
            self.writer,
            scalars=self.scalars,
            key=self.key,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class TableCallback(BaseCallback):
    r"""Callback to log a table to disk."""

    table: DataFrame

    _: KW_ONLY

    writer: SummaryWriter | Path

    # Optional Parameters
    options: Optional[dict[str, Any]] = None
    filetype: str = "parquet"
    name: str = "tables"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, step: int, /, **_: Any) -> None:
        log_table(
            step,
            self.writer,
            table=self.table,
            options=self.options,
            filetype=self.filetype,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class PlotCallback(BaseCallback):
    r"""Callback to log a plot to tensorboard."""

    plot_fn: Callable[Concatenate[int, ...], Figure]

    _: KW_ONLY

    plot_kwargs: Mapping[str, Any] = EMPTY_MAP
    rasterization_options: Mapping[str, Any] = EMPTY_MAP

    writer: SummaryWriter
    name: str = "plot"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, step: int, /, **_: Any) -> None:
        try:
            fig = self.plot_fn(step, **self.plot_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to create plot[{step}]") from exc

        log_plot(
            step,
            self.writer,
            fig=fig,
            rasterization_options=self.rasterization_options,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )
