"""Callback utilities for logging."""

__all__ = [
    # ABCs & Protocols
    "Callback",
    "BaseCallback",
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
    # Functions
    "is_callback",
]

import inspect
import logging
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, MutableSequence, Sequence
from dataclasses import KW_ONLY, dataclass, field
from functools import wraps
from itertools import chain
from pathlib import Path

import torch
import yaml
from pandas import DataFrame, MultiIndex
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from typing_extensions import (
    Any,
    ClassVar,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    TypeGuard,
    TypeVar,
    overload,
    runtime_checkable,
)

from tsdm.logutils.logfuncs import (
    TargetsAndPredics,
    compute_metrics,
    log_config,
    log_kernel,
    log_lr_scheduler,
    log_metrics,
    log_model,
    log_optimizer,
    log_table,
    log_values,
    make_checkpoint,
)
from tsdm.metrics import Metric
from tsdm.models import Model
from tsdm.optimizers import Optimizer
from tsdm.types.aliases import JSON, PathLike
from tsdm.utils.funcutils import get_mandatory_kwargs
from tsdm.utils.pprint import repr_object

P = ParamSpec("P")


@runtime_checkable
class Callback(Protocol[P]):
    """Protocol for callbacks.

    Callbacks should be initialized with mutable objects that are being logged.
    every time the callback is called, it should log the current state of the
    mutable object.
    """

    # required_kwargs: ClassVar[set[str]]
    # """The required kwargs for the callback."""

    # @property
    # def frequency(self) -> int:
    #     """The frequency at which the callback is called."""

    @property
    @abstractmethod
    def required_kwargs(self) -> set[str]:
        """The required kwargs for the callback."""
        ...

    @abstractmethod
    def __call__(self, i: int, /, **kwargs: P.kwargs) -> None:  # pyright: ignore
        """Log something at time index i."""
        ...


# NOTE: overloads from MutableSequence.
# NOTE: Use PEP 695: Type Parameter Syntax in 3.12
CB = TypeVar("CB", bound=Callback)


@overload
def is_callback(obj: CB, /) -> TypeGuard[CB]: ...
@overload
def is_callback(obj: object, /) -> TypeGuard[Callback]: ...
def is_callback(obj, /):
    """Check if the function is a callback."""
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


@dataclass(repr=False)
class BaseCallback(Callback[P]):
    """Base class for callbacks."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    """Logger for the class."""

    # required_kwargs: ClassVar[set[str]]
    # """The required kwargs for the callback."""

    _: KW_ONLY

    frequency: int = 1
    """The frequency at which the callback is called."""

    # @property
    # def required_kwargs(self) -> set[str]:
    #     """The required kwargs for the callback."""
    #     return get_mandatory_kwargs(self.callback)

    @property
    def required_kwargs(self) -> set[str]:
        """The required kwargs for the callback."""
        return get_mandatory_kwargs(self.callback)

    def __init_subclass__(cls) -> None:
        """Automatically set the required kwargs for the callback."""
        super().__init_subclass__()  # Important!

        @wraps(cls.callback)
        def __call__(self, i: int, /, **state_dict: P.kwargs) -> None:
            """Log something at the end of a batch/epoch."""
            if i % self.frequency == 0:
                self.callback(i, **state_dict)

        cls.__call__ = __call__  # type: ignore[method-assign]

    @abstractmethod
    def callback(self, i: int, /, **state_dict: P.kwargs) -> None:  # pyright: ignore
        """Log something at the end of a batch/epoch."""

    def __call__(self, i: int, /, **state_dict: P.kwargs) -> None:  # pyright: ignore
        """Log something at the end of a batch/epoch."""

    def __repr__(self) -> str:
        """Return a string representation of the callback."""
        return repr_object(self)


class CallbackList(BaseCallback, MutableSequence[CB]):
    """Callback to log multiple callbacks."""

    callbacks: list[CB]
    """The callbacks to log."""

    def insert(self, index: int, value: CB) -> None:
        self.callbacks.insert(index, value)

    @overload
    def __getitem__(self, index: int) -> CB: ...
    @overload
    def __getitem__(self, index: slice) -> MutableSequence[CB]: ...
    def __getitem__(self, index):
        return self.callbacks[index]

    @overload
    def __setitem__(self, index: int, value: CB) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[CB]) -> None: ...
    def __setitem__(self, index, value):
        self.callbacks[index] = value

    @overload
    def __delitem__(self, index: int) -> None: ...
    @overload
    def __delitem__(self, index: slice) -> None: ...
    def __delitem__(self, index):
        del self.callbacks[index]

    def __len__(self) -> int:
        return len(self.callbacks)

    @property
    def required_kwargs(self) -> set[str]:
        """The required kwargs for the callback."""
        # result: set[str] = set()
        return set().union(
            *(
                (
                    callback.required_kwargs
                    if hasattr(callback, "required_kwargs")
                    else get_mandatory_kwargs(callback)
                )
                for callback in self.callbacks
            )
        )

    def callback(self, i: int, /, **state_dict: Any) -> None:
        """Log something at the end of a batch/epoch."""
        for callback in self.callbacks:
            callback(i, **state_dict)


@dataclass
class ConfigCallback(BaseCallback):
    """Callback to log the config to tensorboard."""

    config: JSON
    writer: SummaryWriter | Path

    _: KW_ONLY

    fmt: Literal["json", "yaml", "toml"] = "yaml"
    name: str = "config"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **_: Any) -> None:
        """Log the config to tensorboard."""
        log_config(
            i,
            config=self.config,
            writer=self.writer,
            fmt=self.fmt,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class WrapCallback(BaseCallback):
    """Wraps callable as a callback."""

    func: Callable[..., None]

    def callback(self, i: int, /, **state_dict: Any) -> None:
        self.func(i, **state_dict)


@dataclass
class EvaluationCallback(BaseCallback):
    """Callback to log evaluation metrics to tensorboard."""

    _: KW_ONLY

    dataloaders: Mapping[str, DataLoader]
    metrics: (
        Sequence[str | Metric | type[Metric]]
        | Mapping[str, str | Metric | type[Metric]]
    )
    model: Model
    predict_fn: Callable[..., tuple[Tensor, Tensor]]
    writer: SummaryWriter

    history: DataFrame = NotImplemented
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    best_epoch: DataFrame = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the callback."""
        if self.history is NotImplemented:
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

    def callback(self, i: int, /, **_: Any) -> None:
        # update the history
        self.compute_results(i)
        self.compute_best_epoch(i)

        for key in self.dataloaders:
            log_values(
                i,
                scalars=self.history.loc[i, key].to_dict(),
                writer=self.writer,
                key=key,
                name=self.name,
                prefix=self.prefix,
                postfix=self.postfix,
            )
            log_values(
                i,
                scalars=self.best_epoch.loc[i, key].to_dict(),
                writer=self.writer,
                key=key,
                name="score",
                prefix=self.prefix,
                postfix=self.postfix,
            )

        path = Path(self.writer.log_dir) / f"history-{i}.parquet"
        self.history.to_parquet(path)

    def compute_results(self, i: int) -> None:
        """Return the results at the minimal validation loss."""
        for key, dataloader in self.dataloaders.items():
            result = self.get_all_predictions(dataloader)
            scalars = compute_metrics(
                self.metrics, targets=result.targets, predics=result.predics
            )
            for metric, value in scalars.items():
                self.history.loc[i, (key, metric)] = value.cpu().item()

    def compute_best_epoch(self, i: int) -> None:
        """Compute the best epoch for the given index."""
        # unoptimized...
        mask = self.history.index <= i
        best_epochs = (
            self.history.loc[mask, self._val_key]
            .rolling(5, min_periods=1)
            .mean()
            .idxmin()
        )
        for key in self.dataloaders:
            for metric in self.metrics:
                best_value = self.history.loc[best_epochs[metric], (key, metric)]
                self.best_epoch.loc[i, (key, metric)] = best_value

    @torch.no_grad()
    def get_all_predictions(self, dataloader: DataLoader) -> TargetsAndPredics:
        """Return the targets and predictions for the given dataloader."""
        targets_list: list[Tensor] = []
        predics_list: list[Tensor] = []

        for batch in tqdm(dataloader, desc="Evaluation", leave=False):
            target, predic = self.predict_fn(self.model, batch)
            targets_list.append(target)
            predics_list.append(predic)

        targets = torch.nn.utils.rnn.pad_sequence(
            chain.from_iterable(targets_list),  # type: ignore[arg-type]
            batch_first=True,
            padding_value=torch.nan,
        ).squeeze()

        predics = torch.nn.utils.rnn.pad_sequence(
            chain.from_iterable(predics_list),  # type: ignore[arg-type]
            batch_first=True,
            padding_value=torch.nan,
        ).squeeze()

        return TargetsAndPredics(targets=targets, predics=predics)


# @dataclass
# class HParamCallback(BaseCallback):
#     """Callback to log hyperparameters to tensorboard."""
#
#     hparam_dict: dict[str, Any]
#     writer: SummaryWriter
#
#     _: KW_ONLY
#
#     def callback(self, i: int | str, /, **objects: Any) -> None:
#         """Log the test-error selected by validation-error."""
#
#         # add postfix
#         test_scores = {f"metrics:hparam/{k}": v for k, v in scores["test"].items()}
#
#         self.writer.add_hparams(
#             hparam_dict=self.hparam_dict, metric_dict=test_scores, run_name="hparam"
#         )
#         print(f"{test_scores=} achieved by {self.hparam_dict=}")
#
#         # # FIXME: https://github.com/pytorch/pytorch/issues/32651
#         # # os.path.dirname(os.path.realpath(__file__)) + os.sep + log_path
#         # for files in (self.log_dir / "hparam").iterdir():
#         #     shutil.move(files, self.log_dir)
#         # (self.log_dir / "hparam").rmdir()


@dataclass
class CheckpointCallback(BaseCallback):
    """Callback to save checkpoints."""

    objects: Mapping[str, object]
    path: PathLike

    _: KW_ONLY

    def callback(self, i: int, /, **_: Any) -> None:
        make_checkpoint(i, self.objects, path=self.path)


@dataclass
class HParamCallback(BaseCallback):
    """Callback to log hyperparameters to tensorboard."""

    history: DataFrame
    hparam_dict: JSON
    results_dir: PathLike

    _: KW_ONLY

    writer: SummaryWriter

    def callback(self, i: int, /, **_: Any) -> None:
        best_epochs = self.history.rolling(5, center=True).mean().idxmin()

        scores: dict[str, dict[str, float]] = {
            split: {
                metric: float(self.history.loc[idx, (split, metric)])
                for metric, idx in best_epochs["valid"].items()
            }
            for split in ("train", "valid", "test")
        }

        metric_dict = {f"metrics:hparam/{k}": v for k, v in scores["test"].items()}
        run_name = Path(self.writer.log_dir).absolute()

        with open(run_name / f"{i}.yaml", "w", encoding="utf8") as file:
            file.write(yaml.dump(scores))

        self.writer.add_hparams(
            hparam_dict=self.hparam_dict, metric_dict=metric_dict, run_name=run_name
        )
        print(f"Scores {metric_dict=} achieved by {self.hparam_dict=}")


@dataclass
class KernelCallback(BaseCallback):
    """Callback to log kernel information to tensorboard."""

    kernel: Tensor
    writer: SummaryWriter

    _: KW_ONLY

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

    def callback(self, i: int, /, **_: Any) -> None:
        log_kernel(
            i,
            kernel=self.kernel,
            writer=self.writer,
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
    """Callback to log learning rate information to tensorboard."""

    lr_scheduler: LRScheduler
    writer: SummaryWriter

    _: KW_ONLY

    name: str = "lr_scheduler"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **_: Any) -> None:
        log_lr_scheduler(
            i,
            lr_scheduler=self.lr_scheduler,
            writer=self.writer,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class MetricsCallback(BaseCallback):
    """Callback to log multiple metrics to tensorboard."""

    metrics: (
        Sequence[str | Metric | type[Metric]]
        | Mapping[str, str | Metric | type[Metric]]
    )
    writer: SummaryWriter

    _: KW_ONLY

    key: str = ""
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    def callback(  # type: ignore[override]
        self, i: int, /, *, targets: Tensor, predics: Tensor, **_: Any
    ) -> None:
        log_metrics(
            i,
            metrics=self.metrics,
            writer=self.writer,
            targets=targets,
            predics=predics,
            key=self.key,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class ModelCallback(BaseCallback):
    """Callback to log model information to tensorboard."""

    model: Model
    writer: SummaryWriter

    _: KW_ONLY

    log_histograms: bool = False
    log_norms: bool = True
    name: str = "model"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **state_dict: Any) -> None:
        if "model" in state_dict:
            self.model = state_dict["model"]
        log_model(
            i,
            model=self.model,
            writer=self.writer,
            log_histograms=self.log_histograms,
            log_norms=self.log_norms,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class OptimizerCallback(BaseCallback):
    """Callback to log optimizer information to tensorboard."""

    optimizer: Optimizer
    writer: SummaryWriter

    _: KW_ONLY

    log_histograms: bool = False
    log_norms: bool = True
    name: str = "optimizer"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **state_dict: Any) -> None:
        if "optimizer" in state_dict:
            self.optimizer = state_dict["optimizer"]
        log_optimizer(
            i,
            optimizer=self.optimizer,
            writer=self.writer,
            log_histograms=self.log_histograms,
            log_norms=self.log_norms,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class ScalarsCallback(BaseCallback):
    """Callback to log multiple values to tensorboard."""

    scalars: dict[str, Tensor]
    writer: SummaryWriter

    _: KW_ONLY

    key: str = ""
    name: str = "metrics"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **state_dict: Any) -> None:
        for key, value in state_dict.items():
            if key in self.scalars:
                self.scalars[key] = value
        log_values(
            i,
            scalars=self.scalars,
            writer=self.writer,
            key=self.key,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )


@dataclass
class TableCallback(BaseCallback):
    """Callback to log a table to disk."""

    table: DataFrame
    writer: SummaryWriter | Path

    _: KW_ONLY

    options: Optional[dict[str, Any]] = None
    filetype: str = "parquet"
    name: str = "tables"
    prefix: str = ""
    postfix: str = ""

    def callback(self, i: int, /, **state_dict: Any) -> None:
        if (
            hasattr(self.table, "name")
            and isinstance(self.table.name, str)
            and self.table.name in state_dict
        ):
            self.table = state_dict[self.table.name]

        log_table(
            i,
            table=self.table,
            writer=self.writer,
            options=self.options,
            filetype=self.filetype,
            name=self.name,
            prefix=self.prefix,
            postfix=self.postfix,
        )
