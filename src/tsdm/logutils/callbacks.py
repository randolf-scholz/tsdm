r"""Callback utilities for logging.

Callbacks hold the objects or callables needed to log the current state.
Most callback classes take the logged object as a positional constructor argument:

1. A mutable object, such as a `DataFrame` or `nn.Module`, whose state changes over time.
2. A callable that receives a step index and any additional keyword arguments.

`WrapCallback` invokes a callable for its side effects; `PlotCallback` invokes
a callable that returns a figure. Use the `callback(step, **state_dict)` entry
point to apply frequency checks. Calling `__call__` directly bypasses the
frequency check supplied by `BaseCallback`.

Fields may generally be reassigned between calls. `Final` annotations mark
bindings tied to initialized schemas, shared table identities, or persisted
selection state. These are static typing constraints; referenced objects remain
mutable. In particular, update shared histories in place and keep evaluation
split and metric names unchanged. Execution frequencies remain mutable.
"""

__all__ = [
    # ABCs & Protocols & Structural classes
    "BaseCallback",
    "Callback",
    "CallbackSequence",
    "CallbackList",
    "BestEpochTracker",
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
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableSequence
from dataclasses import KW_ONLY, dataclass, field
from itertools import chain
from math import nan
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Concatenate,
    Final,
    Literal,
    Protocol,
    Self,
    SupportsIndex,
    TypeIs,
    final,
    overload,
    runtime_checkable,
)

import polars as pl
import torch
import yaml
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from tsdm.constants import EMPTY_MAP
from tsdm.losses import Loss
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
    r"""Base class with a mutable, keyword-only execution frequency.

    Subclasses inherit `frequency` in their generated dataclass constructors.
    Individual subclasses mark initialization-dependent bindings with `Final`.
    """

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

    writer: SummaryWriter | FilePath

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


type MetricScores = Mapping[str, Mapping[str, float]]


@dataclass(slots=True)
class BestEpochTracker(BaseCallback):
    r"""Select one winning raw row from a shared, append-only history table.

    `history` is retained by reference. The caller must supply a valid table:
    `step` is integer, non-null, unique, and strictly increasing; `selection_column`
    and every other tracked metric are numeric. `best_step` is reserved for output.
    These table invariants are assumed, not validated. Append new evaluations
    in place and keep the schema and existing rows unchanged.
    All owned DataFrames likewise retain their identity through in-place appends.
    Selection settings and the writer are fixed: changing them would mix
    incompatible selection criteria or redirect CSV appends away from the
    header created during initialization. Logging names are also fixed;
    the inherited execution frequency remains mutable.

    Calling with a step label processes all unseen rows through that exact step;
    row positions are never interpreted as step labels. Repeated calls for an
    already processed step are no-ops. Unknown steps raise `KeyError`. This also
    supports pre-existing tables and tracking less often than evaluation.

    Selection defaults to unsmoothed values. Larger windows count evaluations,
    not step-label distance. Nonfinite values occupy window positions but are
    excluded from means; a nonfinite current value cannot win. Ties keep the
    earliest winner. All reported scores are copied from the winning raw row.
    Once a winner exists, `best_step` is automatically nondecreasing: it either
    stays unchanged or advances to the newer step on strict improvement.

    `processed_history` stores each step's processed selection value. `best_history`
    matches the raw schema, with `step` first and an additional `best_step` column.
    Before a winner exists, floating scores are NaN and other scores are null.
    An optional writer enables best-score logging and creates `best_history.csv`
    with its header during initialization. Persistence and logging precede
    in-memory appends, so failed CSV writes leave the shared tables unchanged.
    """

    history: Final[pl.DataFrame]

    _: KW_ONLY

    selection_column: Final[str]
    mode: Final[Literal["min", "max"]] = "min"
    window: Final[int] = 1
    min_periods: Final[int] = 1
    writer: Final[SummaryWriter | None] = None
    name: Final[str] = "score"
    prefix: Final[str] = ""
    postfix: Final[str] = ""

    best_history: Final[pl.DataFrame] = field(init=False)
    processed_history: Final[pl.DataFrame] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        r"""Prepare stable output tables for the supplied history."""
        schema = self.history.schema
        if self.window < 1 or not 1 <= self.min_periods <= self.window:
            raise ValueError("Require window >= 1 and 1 <= min_periods <= window.")
        if self.mode not in {"min", "max"}:
            raise ValueError(f"Unknown selection mode: {self.mode}.")
        self.processed_history = pl.DataFrame(  # type: ignore
            schema={"step": schema["step"], self.selection_column: pl.Float64()}
        )
        self.best_history = pl.DataFrame(  # type: ignore
            schema={
                "step": schema["step"],
                **{
                    column: dtype
                    for column, dtype in schema.items()
                    if column != "step"
                },
                "best_step": schema["step"],
            }
        )

        # initialize the best_history.csv file if a writer is provided
        if self.writer is not None:
            path = Path(self.writer.log_dir) / "best_history.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="") as file:
                self.best_history.write_csv(file)

    def get_step_index(self, step: int, /) -> int:
        r"""Return the row position of an exact step label in the sorted history.

        Raises:
            KeyError: If the history contains no evaluation with this step label.
        """
        steps = self.history["step"]
        position = steps.search_sorted(step)
        if position == len(steps) or steps[position] != step:
            raise KeyError(f"No evaluation for step {step}.")
        return position

    @property
    def best(self) -> pl.DataFrame:
        r"""Return the latest winning-row result without retaining another table."""
        return self.best_history.tail(1)

    @property
    def selection_value(self) -> float | None:
        r"""Look up the winning processed value using the latest `best_step`."""
        if (best_step := self.best_history["best_step"].last()) is None:
            return None

        # Processed history mirrors the shared history's prefix, so indices agree.
        position = self.get_step_index(int(best_step))  # type: ignore
        return self.processed_history[self.selection_column].item(position)

    def __call__(self, step: int, /, **_: Any) -> None:
        r"""Process unseen evaluations through the row whose step equals `step`."""
        processed_rows = self.processed_history.height
        target = self.get_step_index(step)

        # The output row count is also the cursor. Catch up skipped evaluations
        # individually, so each receives its own bounded smoothing window.
        for position in range(processed_rows, target + 1):
            start = max(0, position - self.window + 1)
            recent = self.history.slice(start, position - start + 1)
            processed, selected = self._process(recent)

            # Persist before mutating the shared table so failed writes can be retried.
            if self.writer is not None:
                path = Path(self.writer.log_dir) / "best_history.csv"
                with path.open("a", encoding="utf-8", newline="") as file:
                    selected.write_csv(file, include_header=False)

                scores = selected.drop("step", "best_step").row(0, named=True)
                for column, value in scores.items():
                    split, metric = (
                        column.split("/", maxsplit=1) if "/" in column else ("", column)
                    )
                    log_values(
                        selected["step"].item(),
                        self.writer,
                        scalars={metric: nan if value is None else float(value)},
                        key=split,
                        name=self.name,
                        prefix=self.prefix,
                        postfix=self.postfix,
                    )

            self.processed_history.vstack(processed, in_place=True)
            self.best_history.vstack(selected, in_place=True)

    def _process(self, recent: pl.DataFrame, /) -> tuple[pl.DataFrame, pl.DataFrame]:
        r"""Prepare one processed value and winning row without mutating state."""
        current = recent.tail(1).select(self.best_history.columns[:-1])

        # Start with the previous winner, advancing only the reporting step.
        # Before the first winner, retain the score dtypes with missing values.
        if self.best_history.is_empty():
            selected = current.select(
                "step",
                *(
                    pl.lit(nan if dtype.is_float() else None, dtype=dtype).alias(column)
                    for column, dtype in current.schema.items()
                    if column != "step"
                ),
                pl.lit(None, dtype=current.schema["step"]).alias("best_step"),
            )
        else:
            selected = self.best.with_columns(current["step"])

        # Smooth only the selection column over the supplied trailing window.
        # Nonfinite observations occupy positions but do not enter the mean.
        value = pl.col(self.selection_column)
        processed = recent.select(
            pl.col("step").last(),
            pl.when(value.is_finite().sum() >= self.min_periods)
            .then(value.filter(value.is_finite()).mean().cast(pl.Float64))
            .otherwise(None),
        )
        criterion = processed[self.selection_column].item()

        # Keep the processed value for debugging, even if this raw row cannot win.
        if criterion is None or not current[self.selection_column].is_finite().item():
            return processed, selected

        # Compare against the winning processed value, preserving early ties.
        # On improvement, copy every score from the raw row without smoothing.
        previous = self.selection_value
        if previous is None or (
            criterion < previous if self.mode == "min" else criterion > previous
        ):
            selected = current.with_columns(pl.col("step").alias("best_step"))
        return processed, selected


@dataclass(slots=True)
class EvaluationCallback(BaseCallback):
    r"""Evaluate splits and append raw scores independently of best-epoch selection.

    `history` has one row per evaluation, with `step` first and Float64 score
    columns named `split/metric`. Split names must be valid identifiers. The
    DataFrame reference remains stable; rows are appended in place and step
    labels must strictly increase. Treat existing rows and the schema as read-only.
    Dataloader and metric bindings are fixed because their keys define the schema;
    keep those keys unchanged when mutating the mappings. The writer is fixed
    because its log directory contains the initialized CSV header. The model,
    prediction function, and logging names are also fixed bindings. Model state
    may change in place between evaluations, as may the execution frequency.

    Initialization creates `history.csv` with its header in `writer.log_dir`,
    refusing to overwrite an existing file. Each append writes only the new row.
    Register an optional `BestEpochTracker(history, ...)` after this callback
    to select and persist winning rows.
    """

    model: Final[nn.Module]

    _: KW_ONLY

    predict_fn: Final[Callable[..., tuple[Tensor, Tensor]]]
    dataloaders: Final[Mapping[str, DataLoader]]
    metrics: Final[Mapping[str, Loss]]
    writer: Final[SummaryWriter]
    name: Final[str] = "metrics"
    prefix: Final[str] = ""
    postfix: Final[str] = ""
    history: Final[pl.DataFrame] = field(init=False)

    def __post_init__(self) -> None:
        r"""Validate the configuration and initialize the raw table and CSV header."""
        # Complete configuration checks before creating any files.
        if not self.dataloaders or not self.metrics:
            raise ValueError("Evaluation requires at least one dataloader and metric.")
        if invalid := [split for split in self.dataloaders if not split.isidentifier()]:
            raise ValueError(f"Split names must be valid identifiers, got {invalid}.")

        # initialize the history table with the correct schema
        self.history = pl.DataFrame(  # type: ignore
            schema={
                "step": pl.Int64,
                **{
                    f"{split}/{metric}": pl.Float64
                    for split in self.dataloaders
                    for metric in self.metrics
                },
            }
        )

        # initialize the CSV header in the log directory
        path = Path(self.writer.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "history.csv").open("x", encoding="utf-8", newline="") as file:
            self.history.write_csv(file)

    def __call__(self, step: int, /, **_: Any) -> None:
        scores = self.compute_results()
        self.append_history(step, scores)

        for split, values in scores.items():
            log_values(
                step,
                self.writer,
                scalars=values,
                key=split,
                name=self.name,
                prefix=self.prefix,
                postfix=self.postfix,
            )

    def append_history(self, step: int, scores: MetricScores, /) -> None:
        r"""Persist one complete raw evaluation, then append it in place."""
        if not self.history.is_empty() and step <= (
            last_step := self.history["step"][-1]
        ):
            raise ValueError(f"Expected a step after {last_step}, got {step}.")

        if scores.keys() != self.dataloaders.keys() or any(
            values.keys() != self.metrics.keys() for values in scores.values()
        ):
            raise ValueError(
                "Scores must match all configured dataloaders and metrics."
            )

        row = {
            "step": step,
            **{
                f"{split}/{metric}": value
                for split, values in scores.items()
                for metric, value in values.items()
            },
        }
        batch = pl.DataFrame([row], schema=self.history.schema)

        # Persist before mutating the shared table so failed writes can be retried.
        path = Path(self.writer.log_dir) / "history.csv"
        with path.open("a", encoding="utf-8", newline="") as file:
            batch.write_csv(file, include_header=False)
        self.history.vstack(batch, in_place=True)

    @torch.no_grad()
    def compute_results(self) -> MetricScores:
        r"""Evaluate every split without downcasting scalar metric results."""
        scores: dict[str, dict[str, float]] = {}
        for split, dataloader in self.dataloaders.items():
            result = self.get_all_predictions(dataloader)
            scores[split] = {
                name: float(metric(**result).cpu().item())
                for name, metric in self.metrics.items()
            }
        return scores

    @torch.no_grad()
    def get_all_predictions(
        self, dataloader: DataLoader
    ) -> dict[Literal["targets", "predictions"], Tensor]:
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

        return {"targets": targets, "predictions": predictions}


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
    r"""Log hyperparameters and scores for a given step in a shared history table.

    Retain `history` by reference; append rows in place to keep it current.
    The table must contain unique `step` values; remaining columns contain scores.
    Supply an evaluation's `history` for raw scores or a tracker's `best_history`
    for selected scores. Only `step` is excluded from output; `best_step` is
    retained as an integer when an epoch has been selected. Nulls are preserved
    in YAML and omitted from TensorBoard, which requires numeric scalar values.
    The caller must update the history before reporting. Empty histories raise
    `ValueError`; absent steps raise `polars.exceptions.NoRowsReturnedError`.
    """

    hparam_dict: JSON

    _: KW_ONLY

    history: pl.DataFrame
    writer: SummaryWriter

    def __call__(self, step: int, /, **_: Any) -> None:
        if self.history.is_empty():
            raise ValueError("Hyperparameter logging requires a nonempty history.")
        scores = self.history.row(by_predicate=pl.col("step") == step, named=True)
        scores.pop("step")
        metric_dict = {
            f"metrics:hparam/{column}": value
            for column, value in scores.items()
            if value is not None
        }

        run_name = Path(self.writer.log_dir).absolute()
        with (run_name / f"{step}.yaml").open("w", encoding="utf8") as file:
            yaml.safe_dump(scores, file)

        self.writer.add_hparams(
            hparam_dict=self.hparam_dict,
            metric_dict=metric_dict,
            run_name=run_name,
        )


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

    def __call__(  # ty: ignore[incompatible-method-override]
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

    def __call__(self, step: int, /, **_: Any) -> None:
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

    def __call__(self, step: int, /, **_: Any) -> None:
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
class TableCallback[T](BaseCallback):
    r"""Log a table to disk using the supplied serializer.

    Include the desired suffix in `name`, for example `"tables.parquet"`.
    Configure backend options on `serializer`, for example with
    `functools.partial`.
    """

    table: T

    _: KW_ONLY

    writer: SummaryWriter | FilePath
    serializer: Callable[[T, Path], object]
    name: str

    # Optional Parameters
    prefix: str = ""
    postfix: str = ""

    def __call__(self, step: int, /, **_: Any) -> None:
        log_table(
            step,
            self.writer,
            table=self.table,
            serializer=self.serializer,
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
