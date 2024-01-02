r"""Deprecated Kiwi Task Object."""

__all__ = [
    # Classes
    "KIWI_RUNS_TASK",
    "KIWI_RUNS_GENERATOR",
]

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from itertools import product

import torch
from pandas import DataFrame, MultiIndex, Series
from sklearn.model_selection import ShuffleSplit
from torch import Tensor, jit
from torch.utils.data import DataLoader
from typing_extensions import Any, Literal, NamedTuple, Optional, deprecated

from tsdm.data import MappingDataset, TimeSeriesSampleGenerator
from tsdm.data.timeseries import TimeSeriesDataset
from tsdm.datasets import KiwiRuns, KiwiRunsTSC
from tsdm.encoders import Encoder
from tsdm.metrics import WRMSE
from tsdm.random.samplers import HierarchicalSampler, SequenceSampler
from tsdm.tasks._deprecated import OldBaseTask
from tsdm.utils.pprint import pprint_repr


class KIWI_RUNS_GENERATOR(TimeSeriesSampleGenerator):
    r"""Bioprocess forecasting task using the KIWI-biolab data."""

    targets = ["Base", "DOT", "Glucose", "OD600"]
    observables = [
        "Base",
        "DOT",
        "Glucose",
        "OD600",
        "Acetate",
        "Fluo_GFP",
        "pH",
    ]
    covariates = [
        "Cumulated_feed_volume_glucose",
        "Cumulated_feed_volume_medium",
        "InducerConcentration",
        "StirringSpeed",
        "Flow_Air",
        "Temperature",
        "Probe_Volume",
    ]
    sample_format = ("masked", "masked")

    def __init__(self, **kwargs: Any) -> None:
        ds = KiwiRunsTSC()
        super().__init__(ds, **kwargs)


@pprint_repr
class Sample(NamedTuple):
    r"""A sample of the data."""

    key: tuple[tuple[int, int], slice]
    inputs: tuple[DataFrame, DataFrame]
    targets: float
    originals: Optional[tuple[DataFrame, DataFrame]] = None


@pprint_repr
class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.


@deprecated("outdated task, use tasks.KIWI_Benchmark instead!")
class KIWI_RUNS_TASK(OldBaseTask):
    r"""A collection of bioreactor runs.

    For this task, we do several simplifications:

    - drop run_id 355
    - drop almost all metadata
    - restrict timepoints to start_time & end_time given in metadata.
    - timeseries for each run_id and experiment_id
    - metadata for each run_id and experiment_id

    When first do a train/test split.
    Then the goal is to learn a model in a multitask fashion on all the ts.

    To train, we sample:

    1. random TS from the dataset
    2. random snippets from the sampled TS

    Questions:

    - Should each batch contain only snippets form a single TS, or is there merit to sampling
      snippets from multiple TS in each batch?

    Divide 'Glucose' by 10, 'OD600' by 20, 'DOT' by 100, 'Base' by 200, then use RMSE.
    """

    index: list[tuple[int, str]] = list(product(range(5), ("train", "test")))
    r"""Available index."""
    KeyType = tuple[Literal[0, 1, 2, 3, 4], Literal["train", "test"]]
    r"""Type Hint for Keys."""
    timeseries: DataFrame
    r"""The whole timeseries data."""
    metadata: DataFrame
    r"""The metadata."""
    observation_horizon: int = 96
    r"""The number of datapoints observed during prediction."""
    forecasting_horizon: int = 24
    r"""The number of datapoints the model should forecast."""
    preprocessor: Encoder
    r"""Encoder for the observations."""
    controls: Series
    r"""The control variables."""
    targets: Series
    r"""The target variables."""
    observables: Series
    r"""The observables variables."""

    def __init__(
        self,
        *,
        forecasting_horizon: int = 24,
        observation_horizon: int = 96,
    ):
        super().__init__()
        self.forecasting_horizon = forecasting_horizon
        self.observation_horizon = observation_horizon
        self.horizon = self.observation_horizon + self.forecasting_horizon

        self.timeseries = ts = self.dataset.timeseries
        self.metadata = self.dataset.metadata

        self.targets = targets = Series(["Base", "DOT", "Glucose", "OD600"])
        self.targets.index = self.targets.apply(ts.columns.get_loc)

        self.controls = controls = Series([
            "Cumulated_feed_volume_glucose",
            "Cumulated_feed_volume_medium",
            "InducerConcentration",
            "StirringSpeed",
            "Flow_Air",
            "Temperature",
            "Probe_Volume",
        ])
        controls.index = controls.apply(ts.columns.get_loc)

        self.observables = observables = Series([
            "Base",
            "DOT",
            "Glucose",
            "OD600",
            "Acetate",
            "Fluo_GFP",
            "pH",
        ])
        observables.index = observables.apply(ts.columns.get_loc)

        assert (
            set(controls.values) | set(targets.values) | set(observables.values)
        ) == set(ts.columns)

    @cached_property
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The metric to be used for evaluation."""
        ts = self.timeseries
        weights = DataFrame.from_dict(
            {
                "Base": 200,
                "DOT": 100,
                "Glucose": 10,
                "OD600": 20,
            },
            orient="index",
            columns=["inverse_weight"],
        )
        weights["col_index"] = weights.index.map(lambda x: (ts.columns == x).argmax())
        weights["weight"] = 1 / weights["inverse_weight"]
        weights["normalized"] = weights["weight"] / weights["weight"].sum()
        weights.index.name = "col"
        w = torch.tensor(weights["weight"])
        return jit.script(WRMSE(w))  # pyright: ignore

    @cached_property
    def dataset(self) -> KiwiRuns:
        r"""Return the cached dataset."""
        dataset = KiwiRuns()
        dataset.metadata.drop([482], inplace=True)
        dataset.timeseries.drop([482], inplace=True)
        return dataset

    @cached_property
    def split_idx(self) -> DataFrame:
        r"""Return table with indices for each split."""
        splitter = ShuffleSplit(n_splits=5, random_state=0, test_size=0.25)
        groups = self.metadata.groupby(["color", "run_id"])
        group_idx = groups.ngroup()

        splits = DataFrame(index=self.metadata.index)
        for i, (train, _) in enumerate(splitter.split(groups)):
            splits[i] = group_idx.isin(train).map({False: "test", True: "train"})

        splits.columns.name = "split"
        return splits.astype("string").astype("category")

    @cached_property
    def split_idx_sparse(self) -> DataFrame:
        r"""Return sparse table with indices for each split."""
        df = self.split_idx
        columns = df.columns

        # get categoricals
        categories = {
            col: df[col].astype("category").dtype.categories for col in columns
        }

        if isinstance(df.columns, MultiIndex):
            index_tuples = [
                (*col, cat)
                for col, cats in zip(columns, categories, strict=True)
                for cat in categories[col]
            ]
            names = [*df.columns.names, "partition"]
        else:
            index_tuples = [
                (col, cat)
                for col, cats in zip(columns, categories, strict=True)
                for cat in categories[col]
            ]
            names = [df.columns.name, "partition"]

        new_columns = MultiIndex.from_tuples(index_tuples, names=names)
        result = DataFrame(index=df.index, columns=new_columns, dtype=bool)

        if isinstance(df.columns, MultiIndex):
            for col in new_columns:
                result[col] = df[col[:-1]] == col[-1]
        else:
            for col in new_columns:
                result[col] = df[col[0]] == col[-1]

        return result

    @cached_property
    def splits(self) -> dict[Any, tuple[DataFrame, DataFrame]]:
        r"""Return a subset of the data corresponding to the split."""
        splits = {}
        for key in self.index:
            assert key in self.index, f"Wrong {key=}. Only {self.index} work."
            split, data_part = key

            mask = self.split_idx[split] == data_part
            idx = self.split_idx[split][mask].index
            timeseries = self.timeseries.reset_index(level=2).loc[idx]
            timeseries = timeseries.set_index("measurement_time", append=True)
            metadata = self.metadata.loc[idx]
            splits[key] = (timeseries, metadata)
        return splits

    @cached_property
    def dataloader_kwargs(self) -> dict:
        r"""Return the kwargs for the dataloader."""
        return {
            "batch_size": 1,
            "shuffle": False,
            "sampler": None,
            "batch_sampler": None,
            "num_workers": 0,
            "collate_fn": lambda x: x,
            "pin_memory": False,
            "drop_last": False,
            "timeout": 0,
            "worker_init_fn": None,
            "prefetch_factor": 2,
            "persistent_workers": False,
        }

    def make_dataloader(
        self, key: KeyType, /, *, shuffle: bool = False, **dataloader_kwargs: Any
    ) -> DataLoader:
        r"""Return a dataloader for the given split."""
        # Construct the dataset object
        ts, md = self.splits[key]
        dataset = _Dataset(
            ts,
            md,
            observables=self.observables.index,
            observation_horizon=self.observation_horizon,
            targets=self.targets.index,
        )

        # fmt: off
        DS = MappingDataset({
            idx: TimeSeriesDataset(ts.loc[idx], metadata=md.loc[idx])
            for idx in md.index
        })
        # fmt: on
        # construct the sampler
        subsamplers = {
            key: SequenceSampler(
                ds.timeseries,
                seq_len=self.horizon,
                stride=1,
                shuffle=shuffle,
            )
            for key, ds in DS.items()
        }
        sampler = HierarchicalSampler(DS, subsamplers, shuffle=shuffle)

        # construct the dataloader
        kwargs: dict[str, Any] = {"collate_fn": lambda x: x} | dataloader_kwargs
        return DataLoader(dataset, sampler=sampler, **kwargs)


@dataclass
class _Dataset(torch.utils.data.Dataset):
    timeseries: DataFrame
    metadata: DataFrame

    _: KW_ONLY

    observables: list[str]
    targets: list[str]
    observation_horizon: int

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, item: tuple[tuple[int, int], slice], /) -> Sample:
        r"""Return a sample from the dataset."""
        key, slc = item
        ts = self.timeseries.loc[key].iloc[slc].copy(deep=True)
        md = self.metadata.loc[key].copy(deep=True)
        originals = (ts.copy(deep=True), md.copy(deep=True))
        targets = ts.iloc[self.observation_horizon :, self.targets].copy(deep=True)
        ts.iloc[self.observation_horizon :, self.targets] = float("nan")
        ts.iloc[self.observation_horizon :, self.observables] = float("nan")
        return Sample(key=item, inputs=(ts, md), targets=targets, originals=originals)
