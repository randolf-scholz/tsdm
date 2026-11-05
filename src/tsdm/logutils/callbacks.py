r"""Callback utilities for logging.

Callbacks hold the objects or callables needed to log the current state.
Most callback classes take the logged object as a positional constructor argument:

1. A mutable object, such as a `DataFrame` or `nn.Module`, whose state changes over time.
2. A callable that receives a step index and any additional keyword arguments.

`WrapCallback` invokes a callable for its side effects; `PlotCallback` invokes
a callable that returns a figure. Use the `callback(step, **state_dict)` entry
point to apply frequency checks. Calling `__call__` directly bypasses the
frequency check supplied by `BaseCallback`.
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
    "PlotCallback",
    "ScalarsCallback",
    "TableCallback",
    "WrapCallback",
    # Functions
    "is_callback",
]

import inspect
import logging
import pickle
from abc import abstractmethod
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableSequence,
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
from pandas import DataFrame, MultiIndex, Series
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from tsdm.constants import EMPTY_MAP, UNDEFINED
from tsdm.pprint import pprint_repr, pprint_sequence
from tsdm.types import FilePath
from tsdm.types.nested import JSON
from tsdm.utils.funcutils import get_mandatory_kwargs

from .logfuncs import (
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
from .utils import Loss, TargetsAndPredictions


@runtime_checkable
class Callback(Protocol):
    r"""Protocol for callbacks.

    Callbacks log the current state of stored objects or values supplied as
    keyword arguments. The `callback` method receives the current step and
    applies the callback's execution frequency.
    """

    @property
    def frequency(self) -> int:
        r"""Positive number of steps between executions."""
        return 1

    @property
    @abstractmethod
    def required_kwargs(self) -> set[str]:
        r"""Names of the keyword arguments required to execute the callback."""
        ...

    @abstractmethod
    def callback(self, step: int, /, **state_dict: Any) -> None:
        r"""Log the supplied state at `step` when the execution frequency permits."""
        ...


@runtime_checkable
class CallbackSequence[CallbackT: Callback](Callback, Protocol):
    r"""Protocol for sequences of callbacks.

    Combine the `Callback` interface with mutable sequence operations on
    elements of type `CallbackT`.
    """

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[CallbackT]: ...

    def __contains__(self, value: object, /) -> bool: ...

    @overload
    def __getitem__(self, index: int, /) -> CallbackT: ...

    @overload
    def __getitem__(self, index: slice, /) -> Self: ...

    @overload
    def __setitem__(self, index: int, value: CallbackT, /) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[CallbackT], /) -> None: ...

    @overload
    def __delitem__(self, index: int, /) -> None: ...

    @overload
    def __delitem__(self, index: slice, /) -> None: ...

    def __iadd__(self, values: Iterable[CallbackT], /) -> Self: ...

    def insert(self, index: int, value: CallbackT, /) -> None: ...

    def append(self, value: CallbackT, /) -> None: ...

    def extend(self, values: Iterable[CallbackT], /) -> None: ...

    def pop(self, index: int = -1, /) -> CallbackT: ...

    def clear(self, /) -> None: ...


def is_callback(obj: object, /) -> TypeIs[Callback]:
    r"""Check whether a callable has the expected callback signature.

    The first parameter must be positional-only; all remaining parameters must
    be keyword-only or accept arbitrary keyword arguments. This signature check
    does not verify the other attributes required by the `Callback` protocol.
    A callable whose signature cannot be inspected raises `ValueError`.
    """
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
@dataclass(slots=True)
class BaseCallback:
    r"""Base class for callbacks."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for the class."""

    _: KW_ONLY

    frequency: int = 1
    r"""Positive number of steps between executions; must be nonzero."""

    @property
    def required_kwargs(self) -> set[str]:
        r"""Names of the keyword arguments required by `__call__`."""
        return get_mandatory_kwargs(self.__call__)

    @abstractmethod
    def __call__(self, step: int, /, **state_dict: Any) -> None:
        r"""Execute the callback without applying the frequency check."""
        ...

    @final
    def callback(self, step: int, /, **state_dict: Any) -> None:
        r"""Execute the callback if `step` is a multiple of `frequency`.

        Forward all keyword arguments to `__call__`. A zero frequency raises
        `ZeroDivisionError`; use a positive frequency.
        """
        if step % self.frequency == 0:
            self.LOGGER.debug("Logging at step %d", step)
            self(step, **state_dict)


@pprint_sequence
class CallbackList(MutableSequence[Callback], BaseCallback):
    r"""Execute an ordered, mutable sequence of callbacks."""

    callbacks: list[Callback]
    r"""Callbacks to execute in order, each with its own frequency check."""

    @property
    def required_kwargs(self) -> set[str]:
        r"""Union of the keyword argument names required by the child callbacks."""
        return set().union(*(callback.required_kwargs for callback in self.callbacks))

    def __init__(
        self, iterable: Iterable[Callback] = (), /, *, frequency: int = 1
    ) -> None:
        r"""Store the callbacks and set the execution frequency of the list."""
        super().__init__(frequency=frequency)
        self.callbacks = list(iterable)

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self) -> Iterator[Callback]:
        return iter(self.callbacks)

    @overload
    def __getitem__(self, index: int, /) -> Callback: ...

    @overload
    def __getitem__(self, index: slice, /) -> Self: ...

    def __getitem__(self, index: int | slice, /) -> Callback | Self:
        if isinstance(index, SupportsIndex):
            return self.callbacks[index]
        return self.__class__(self.callbacks[index])

    @overload
    def __setitem__(self, index: int, value: Callback, /) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Callback], /) -> None: ...

    def __setitem__(self, index: int | slice, value: Any, /) -> None:
        self.callbacks[index] = value

    @overload
    def __delitem__(self, index: int, /) -> None: ...

    @overload
    def __delitem__(self, index: slice, /) -> None: ...

    def __delitem__(self, index: int | slice, /) -> None:
        del self.callbacks[index]

    def __call__(self, step: int, /, **state_dict: Any) -> None:
        r"""Dispatch `step` and the supplied state to each child callback."""
        for obj in self.callbacks:
            obj.callback(step, **state_dict)

    def insert(self, index: int, value: Callback) -> None:
        self.callbacks.insert(index, value)


@dataclass(slots=True)
class WrapCallback(BaseCallback):
    r"""Wrap a callable that accepts a step and keyword arguments as a callback."""

    func: Callable[Concatenate[int, ...], None]

    def __call__(self, step: int, /, **state_dict: Any) -> None:
        self.func(step, **state_dict)


@dataclass(slots=True)
class ConfigCallback(BaseCallback):
    r"""Save the configuration to a JSON or YAML file via `log_config`.

    Use `writer.log_dir` or the supplied path as the destination directory.
    TOML output is not implemented.
    """

    config: JSON

    _: KW_ONLY

    writer: SummaryWriter | Path

    # Optional parameters
    fmt: Literal["json", "yaml", "toml"] = "yaml"
    name: str = "config"
    prefix: str = ""
    postfix: str = ""

    def __call__(self, step: int, /, **_: Any) -> None:
        r"""Save the configuration to a file identified by `step`."""
        log_config(
            step,
            self.writer,
            config=self.config,
            fmt=self.fmt,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass(slots=True)
class EvaluationCallback(BaseCallback):
    r"""Evaluate data splits, log metrics to TensorBoard, and save the history.

    Call `predict_fn(model, batch)` for each batch to obtain targets and
    predictions. The history has step indices and `(split, metric)` columns.
    Selecting best epochs requires exactly one split named `val`, `valid`, or
    `validation`, ignoring case. Metrics are treated as losses to minimize.
    """

    model: nn.Module  # mutable

    _: KW_ONLY

    predict_fn: Callable[..., tuple[Tensor, Tensor]]
    dataloaders: Mapping[str, DataLoader]
    metrics: Mapping[str, Loss]
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
        r"""Initialize the history tables and recompute results for existing steps."""
        if self.history is UNDEFINED:
            self.history = DataFrame(
                columns=MultiIndex.from_product([self.dataloaders, self.metrics])
            )
        self.best_epoch = self.history.copy()

        for i in self.history.index:
            self.compute_results(i)

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

    def _get_validation_key(self) -> str:
        # make sure there is exactly one validation split.
        candidate: str | None = None
        for key in self.dataloaders:
            if key.lower() in {"val", "valid", "validation"}:
                if candidate is not None:
                    raise ValueError(
                        f"Found multiple validation splits: {candidate} and {key}!"
                    )
                candidate = key
        if candidate is None:
            raise ValueError("Could not find a validation split!")
        return candidate

    @torch.no_grad()
    def compute_results(self, step: int) -> None:
        r"""Evaluate every data split and store its metrics in `history` at `step`."""
        for key, dataloader in self.dataloaders.items():
            result = self.get_all_predictions(dataloader)
            scalars = {
                key: metric(predictions=result.predictions, targets=result.targets)
                for key, metric in self.metrics.items()
            }
            for metric, value in scalars.items():
                self.history.loc[step, (key, metric)] = value.cpu().item()

    def compute_best_epoch(self, step: int) -> None:
        r"""Store split scores from the best validation epoch for each metric.

        Consider history rows with indices at most `step`. Choose an epoch for
        each metric by minimizing its trailing five-row validation mean, allowing
        windows with a single observation. Store the unsmoothed scores from those
        epochs across all splits in `best_epoch` at `step`.
        """
        # unoptimized...
        mask = self.history.index <= step
        validation_key = self._get_validation_key()
        best_epochs = (
            self.history.loc[mask, validation_key]
            .rolling(5, min_periods=1)
            .mean()
            .idxmin()
        )
        for key in self.dataloaders:
            for metric in self.metrics:
                best_value = self.history.loc[best_epochs[metric], (key, metric)]
                self.best_epoch.loc[step, (key, metric)] = best_value

    @torch.no_grad()
    def get_all_predictions(self, dataloader: DataLoader) -> TargetsAndPredictions:
        r"""Collect targets and predictions, pad with NaNs, and squeeze singleton axes.

        `predict_fn(model, batch)` must return targets and predictions whose
        leading dimension indexes sequences with compatible trailing shapes.
        The dataloader must yield at least one sequence.
        """
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

        predictions = torch.nn.utils.rnn.pad_sequence(
            list(chain.from_iterable(predics_list)),
            batch_first=True,
            padding_value=torch.nan,
        ).squeeze()

        return TargetsAndPredictions(targets=targets, predictions=predictions)


@dataclass(slots=True)
class CheckpointCallback(BaseCallback):
    r"""Callback to save checkpoints."""

    objects: Mapping[str, object]  # mutable objects to snapshot

    _: KW_ONLY

    path: FilePath

    def __call__(self, step: int, /, **_: Any) -> None:
        path = Path(self.path) / f"{step}"
        path.mkdir(parents=True, exist_ok=True)

        for name, obj in self.objects.items():
            match obj:
                case None:
                    pass
                case torch.jit.ScriptModule():
                    torch.jit.save(obj, path / name)
                case nn.Module():
                    torch.save(obj, path / name)
                case Optimizer():
                    torch.save(obj, path / name)
                case LRScheduler():
                    torch.save(obj, path / name)
                case dict() | list() | tuple() | set() | str() | int() | float() | None:
                    path /= f"{name}.yaml"
                    with path.open("w", encoding="utf8") as file:
                        yaml.safe_dump(obj, file)
                case _:
                    path /= f"{name}.pickle"
                    with path.open("wb") as file:
                        pickle.dump(obj, file)


@dataclass(slots=True)
class HParamCallback(BaseCallback):
    r"""Callback to log hyperparameters to TensorBoard."""

    hparam_dict: JSON

    _: KW_ONLY

    history: DataFrame
    r"""Metric history with `(split, metric)` columns for `train`, `valid`, and `test`."""
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
        r"""Scores at epochs minimizing centered five-row validation means.

        Select an epoch independently for each metric, then return that epoch's
        unsmoothed scores for `train`, `valid`, and `test`. The rolling mean uses
        full windows, so at least five history rows with valid values are needed
        for each metric to select an epoch.
        """
        best_epochs: Series = self.history.rolling(5, center=True).mean().idxmin()

        return {
            split: {
                str(metric): float(self.history.loc[idx, (split, metric)])
                for metric, idx in best_epochs["valid"].items()
            }
            for split in ("train", "valid", "test")
        }


@dataclass(slots=True)
class KernelCallback(BaseCallback):
    r"""Callback to log kernel information to TensorBoard."""

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


@dataclass(slots=True)
class LRSchedulerCallback(BaseCallback):
    r"""Callback to log learning rate information to TensorBoard."""

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


@dataclass(slots=True)
class MetricsCallback(BaseCallback):
    r"""Callback to log multiple metrics to TensorBoard."""

    metrics: Mapping[str, Loss]

    _: KW_ONLY

    writer: SummaryWriter

    # Optional Parameters
    key: str = ""
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    def __call__(  # ty: ignore[invalid-method-override]
        self, step: int, /, *, targets: Tensor, predictions: Tensor, **_: Any
    ) -> None:
        log_metrics(
            step,
            self.writer,
            metrics=self.metrics,
            predictions=predictions,
            targets=targets,
            key=self.key,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass(slots=True)
class ModelCallback(BaseCallback):
    r"""Callback to log model information to TensorBoard."""

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


@dataclass(slots=True)
class OptimizerCallback(BaseCallback):
    r"""Callback to log optimizer information to TensorBoard."""

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


@dataclass(slots=True)
class ScalarsCallback(BaseCallback):
    r"""Callback to log multiple scalar values to TensorBoard."""

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


@dataclass(slots=True)
class TableCallback(BaseCallback):
    r"""Callback to log a table to disk."""

    table: Any

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


@dataclass(slots=True)
class PlotCallback(BaseCallback):
    r"""Call `plot_fn(step, **plot_kwargs)` and log the returned figure to TensorBoard."""

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
