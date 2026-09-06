"""Backend-agnostic contracts for reproducible forecasting tasks.

A forecasting task combines a raw time-series dataset with a prescribed splitting,
sampling, batching, and evaluation procedure. The task defines these operations through
``make_*`` factory methods; concrete subclasses decide whether they use pandas, Polars,
NumPy, Torch, or another backend.
"""

__all__ = [
    "ForecastingMetric",
    "ForecastingTask",
    "RawTimeSeries",
]

from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from tsdm.timeseries.abstract import SplitTimeData, TimeSeries, TimeSeriesCollection
from tsdm.types.protocols import SupportsGetItem

type RawTimeSeries[TableT, TimeT, SeriesID] = (
    TimeSeries[TableT, TimeT] | TimeSeriesCollection[SeriesID, TableT]
)
r"""Raw dataset accepted by a forecasting task."""


class ForecastingMetric[PredictionT, TargetT, ScoreT](Protocol):
    r"""Metric used to compare forecasts with target values.

    All metrics follow a minimization convention: a lower score is always considered
    better. Metrics that are conventionally maximized should negate or otherwise
    transform their result.
    """

    def __call__(self, *, predictions: PredictionT, targets: TargetT) -> ScoreT:
        r"""Evaluate predictions against targets."""
        ...


@runtime_checkable
class ForecastingTask[
    SplitID: Hashable,
    SelectionT,
    DatasetT: RawTimeSeries[Any, Any, Any],
    SampleID,
    SampleT,
    BatchT,
    PredictionT,
    ScoreT,
](Protocol):
    r"""Backend-agnostic interface for a time-series forecasting benchmark.

    The raw ``dataset`` must satisfy :class:`TimeSeries` or
    :class:`TimeSeriesCollection`. Every generated sample must satisfy
    :class:`SeparateTimeSample`. A batch follows the same protocol, typically with
    extra leading batch dimensions and possibly a different array backend.

    ``SelectionT`` describes how a split is selected from the raw dataset. For example,
    it may be a pandas mask, a Polars expression, a collection of entity identifiers,
    or a backend-independent interval object. The task subclass owns its interpretation.

    Test metrics follow a minimization convention: lower values are always better.
    """

    # fmt: off
    @property
    def split_ids(self) -> Sequence[SplitID]: ...
    @property
    def dataset(self) -> DatasetT: ...
    @property
    def folds(self) -> Mapping[SplitID, SelectionT]: ...
    @property
    def splits(self) -> Mapping[SplitID, DatasetT]: ...
    @property
    def generators(self) -> Mapping[SplitID, SupportsGetItem[SampleID, SampleT]]: ...
    @property
    def samplers(self) -> Mapping[SplitID, Iterable[SampleID]]: ...
    @property
    def batchers(self) -> Mapping[SplitID, Callable[[Sequence[SampleT]], BatchT]]: ...
    @property
    def dataloaders(self) -> Mapping[SplitID, Iterable[BatchT]]: ...
    @property
    def test_metrics(self) -> Mapping[SplitID, ForecastingMetric[PredictionT, BatchT, ScoreT]]: ...
    # fmt: on

    @abstractmethod
    def make_folds(self, /) -> Mapping[SplitID, SelectionT]:
        r"""Create the canonical split selections."""
        ...

    @abstractmethod
    def make_split(self, key: SplitID, /) -> DatasetT:
        r"""Materialize one split from the raw dataset."""
        ...

    @abstractmethod
    def make_generator(
        self, key: SplitID, /
    ) -> SupportsGetItem[SampleID, SplitTimeData[SampleT]]:
        r"""Create the sample generator for one split."""
        ...

    @abstractmethod
    def make_sampler(self, key: SplitID, /) -> Iterable[SampleID]:
        r"""Create the canonical sample-key iterable for one split."""
        ...

    @abstractmethod
    def make_batcher(
        self, key: SplitID, /
    ) -> Callable[
        [Sequence[SplitTimeData[SampleT]]],
        SplitTimeData[BatchT],
    ]:
        r"""Create the split-aware sample-to-batch conversion.

        The batcher may bind preprocessing state fitted on the corresponding training
        split so that validation and test data use the same transformation.
        """
        ...

    @abstractmethod
    def make_dataloader(self, key: SplitID, /) -> Iterable[SplitTimeData[BatchT]]:
        r"""Create the model-facing dataloader for one split."""
        ...

    @abstractmethod
    def make_test_metric(
        self, key: SplitID, /
    ) -> ForecastingMetric[PredictionT, BatchT, ScoreT]:
        r"""Create the canonical minimization metric for one split."""
        ...
