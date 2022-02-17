r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "KIWI_FINAL_PRODUCT",
]
import logging
import os
from functools import cached_property
from itertools import product
from typing import Any, Callable, Literal, NamedTuple, Optional

import pandas as pd
import torch
from pandas import DataFrame, MultiIndex, Series, Timedelta, Timestamp
from sklearn.model_selection import ShuffleSplit
from torch import Tensor, jit
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from tsdm.datasets import KIWI_RUNS
from tsdm.datasets.torch import MappingDataset, TimeSeriesDataset
from tsdm.random.samplers import HierarchicalSampler, IntervalSampler
from tsdm.tasks.base import BaseTask
from tsdm.util.decorators import IterItems
from tsdm.util.strings import repr_array, repr_mapping

__logger__ = logging.getLogger(__name__)


class Batch(NamedTuple):
    r"""Represents a batch of elements."""

    index: Any
    timeseries: Any
    metadata: Any
    targets: Any

    def __repr__(self):
        return repr_mapping(
            self._asdict(), title=self.__class__.__name__, repr_fun=repr_array
        )


class Split(NamedTuple):
    r"""Represents a data split."""

    number: int
    partition: Literal["train", "test"]
    timeseries: Tensor
    metadata: Tensor

    def __repr__(self):
        return repr_mapping(
            self._asdict(), title=self.__class__.__name__, repr_fun=repr_array
        )


def get_induction_time(s: Series) -> Timestamp:
    r"""Compute the induction time."""
    # s = ts.loc[run_id, exp_id]
    inducer = s["InducerConcentration"]
    total_induction = inducer[-1] - inducer[0]

    if pd.isna(total_induction) or total_induction == 0:
        return pd.NA

    diff = inducer.diff()
    mask = pd.notna(diff) & (diff != 0.0)
    inductions = inducer[mask]
    assert len(inductions) == 1, "Multiple Inductions occur!"
    return inductions.first_valid_index()


def get_final_product(s: Series, target: str) -> Timestamp:
    r"""Compute the final product time."""
    # Final and target times
    targets = s[target]
    mask = pd.notna(targets)
    targets = targets[mask]
    assert len(targets) >= 1, f"not enough target observations {targets}"
    return targets.index[-1]


def get_time_table(
    ts: DataFrame, target: str = "Fluo_GFP", t_min: str = "0.6h"
) -> DataFrame:
    r"""Compute the induction time and final product time for each run and experiment."""
    columns = [
        # "slice",
        "t_min",
        "t_induction",
        "t_max",
        "t_target",
    ]
    index = ts.reset_index(level=[2]).index.unique()
    df = DataFrame(index=index, columns=columns)

    min_wait = Timedelta(t_min)

    for idx, slc in ts.groupby(level=[0, 1]):
        slc = slc.reset_index(level=[0, 1], drop=True)
        t_induction = get_induction_time(slc)
        t_target = get_final_product(slc, target=target)
        if pd.isna(t_induction):
            __logger__.info("%s: no t_induction!", idx)
            t_max = get_final_product(slc.loc[slc.index < t_target], target=target)
            assert t_max < t_target
        else:
            assert t_induction < t_target, f"{t_induction=} after {t_target}!"
            t_max = t_induction
        df.loc[idx, "t_max"] = t_max

        df.loc[idx, "t_min"] = t_min = slc.index[0] + min_wait
        df.loc[idx, "t_induction"] = t_induction
        df.loc[idx, "t_target"] = t_target
        # df.loc[idx, "slice"] = slice(t_min, t_max)
    return df


class KIWI_FINAL_PRODUCT(BaseTask):
    r"""Forecast the final biomass or product."""

    KeyType = tuple[Literal[0, 1, 2, 3, 4], Literal["train", "test"]]
    r"""Type Hint for Keys"""
    index: list[KeyType] = list(product(range(5), ("train", "test")))  # type: ignore[arg-type]
    r"""Available index."""
    target: Literal["Fluo_GFP", "OD600"]
    r"""The target variables."""
    delta_t: Timedelta
    r"""The time resolution."""
    t_min: Timedelta
    r"""The minimum observation time."""
    timeseries: DataFrame
    r"""The whole timeseries data."""
    metadata: DataFrame
    r"""The metadata."""
    dataloader_kwargs: dict[str, Any]
    r"""The dataloader kwargs."""

    def __init__(
        self,
        target: Literal["Fluo_GFP", "OD600"] = "Fluo_GFP",
        t_min: str = "0.6h",
        delta_t: str = "5m",
        eval_batch_size: int = 128,
        train_batch_size: int = 32,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        cpus: int = (os.cpu_count() or 0) // 2

        self.dataloader_kwargs = {
            "pin_memory": False,
            "num_workers": max(1, min(cpus, 4)),
        }

        # Setup Dataset, drop runs that don't work for this task.
        dataset = self.dataset
        dataset.timeseries = dataset.timeseries.drop([355, 445, 482]).astype(float)
        dataset.metadata = dataset.metadata.drop([355, 445, 482])

        self.timeseries = ts = self.dataset.timeseries
        self.metadata = md = self.dataset.metadata

        # Initialize other attributes
        self.target = target
        self.t_min = t_min
        self.delta_t = Timedelta(delta_t)

        self.eval_batch_size = eval_batch_size
        self.train_batch_size = train_batch_size

        self.final_product_times = get_time_table(
            ts, target=self.target, t_min=self.t_min
        )

        # Compute the final vector
        final_vecs = {}

        for idx in md.index:
            t_target = self.final_product_times.loc[idx, "t_target"]
            final_vecs[(*idx, t_target)] = ts.loc[idx].loc[t_target]

        final_value = DataFrame.from_dict(final_vecs, orient="index")
        final_value.index = final_value.index.set_names(ts.index.names)
        self.final_value = final_value[target]
        self.final_value = self.final_value.reset_index(-1)
        self.final_value = self.final_value.rename(
            columns={"measurement_time": "final_time"}
        )

        self.metadata = self.metadata.join(self.final_value)

        # Construct the dataset object
        TSDs = {}
        for key in self.metadata.index:
            TSDs[key] = TimeSeriesDataset(
                self.timeseries.loc[key],
                metadata=(self.metadata.loc[key], self.final_value.loc[key]),
            )
        DS = MappingDataset(TSDs)
        self.DS = DS

    @cached_property
    def dataset(self) -> KIWI_RUNS:
        r"""Return the cached dataset."""
        return KIWI_RUNS()

    @cached_property
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The target metric."""
        return jit.script(MSELoss())

    @cached_property
    def split_idx(self) -> DataFrame:
        r"""Return table with indices for each split.

        Returns
        -------
        DataFrame
        """
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
        r"""Return sparse table with indices for each split.

        Returns
        -------
        DataFrame[bool]
        """
        df = self.split_idx
        columns = df.columns

        # get categoricals
        categories = {
            col: df[col].astype("category").dtype.categories for col in columns
        }

        if isinstance(df.columns, MultiIndex):
            index_tuples = [
                (*col, cat)
                for col, cats in zip(columns, categories)
                for cat in categories[col]
            ]
            names = df.columns.names + ["partition"]
        else:
            index_tuples = [
                (col, cat)
                for col, cats in zip(columns, categories)
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
    def splits(self) -> dict[KeyType, tuple[DataFrame, DataFrame]]:
        r"""Return a subset of the data corresponding to the split.

        Returns
        -------
        tuple[DataFrame, DataFrame]
        """
        splits = {}
        for key in self.index:
            assert key in self.index, f"Wrong {key=}. Only {self.index} work."
            split, data_part = key

            mask = self.split_idx[split] == data_part
            idx = self.split_idx[split][mask].index
            timeseries = self.timeseries.reset_index(level=2).loc[idx]
            timeseries = timeseries.set_index("measurement_time", append=True)
            metadata = self.metadata.loc[idx]
            splits[key] = timeseries, metadata
            # splits[key] = Split(key[0], key[1], timeseries, metadata)
        return splits

    @staticmethod
    def collate_fn(batch: list[tuple]) -> Batch:
        r"""Batch the sampled items."""
        index = []
        timeseries = []
        metadata = []
        targets = []
        for idx, (ts_data, (md_data, target)) in batch:
            index.append(idx)
            timeseries.append(ts_data)
            metadata.append(md_data)
            targets.append(target)

        return Batch(index, timeseries, metadata, targets)

    def get_dataloader(
        self,
        key: KeyType,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> DataLoader:
        r"""Return a dataloader for the given split.

        Parameters
        ----------
        key: KeyType,
        batch_size: int = 1,
        shuffle: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        kwargs: Any,

        Returns
        -------
        DataLoader
        """
        # Construct the dataset object
        ts, md = self.splits[key]
        TSDs = {}
        for idx in md.index:
            TSDs[idx] = TimeSeriesDataset(
                ts.loc[idx],
                metadata=(md.loc[idx], self.final_value.loc[idx]),
            )
        DS = IterItems(MappingDataset(TSDs))

        # Setup the samplers
        subsamplers = {}
        for idx in TSDs:
            subsampler = IntervalSampler(
                xmin=self.final_product_times.loc[idx, "t_min"],
                xmax=self.final_product_times.loc[idx, "t_max"],
                # offset=t_0,
                deltax=lambda k: k * self.delta_t,
                stride=None,
                shuffle=True,
            )
            subsamplers[idx] = subsampler
        sampler = HierarchicalSampler(TSDs, subsamplers, shuffle=True)

        # Construct the dataloader
        dataloader = DataLoader(
            DS,
            sampler=sampler,
            batch_size=batch_size,
            **(
                self.dataloader_kwargs  # type: ignore[arg-type]
                | {
                    "collate_fn": self.collate_fn,
                }
                | kwargs
            ),
        )
        return dataloader


# # Fit the preprocessor on the training data
# ts_train, md_train = self.splits[(split, "train")]
# preprocessor = deepcopy(self.preprocessor)
# preprocessor.fit(ts_train.reset_index(level=[0, 1], drop=True))
#
# # get the target encoder
# target_idx = self.timeseries.columns.get_loc(self.target)
# target_encoder = (
#     TensorEncoder() @ FloatEncoder() @ preprocessor[-1][target_idx]
# )
#
#
# # Setup the preprocessor
# if preprocessor is None:
#     self.preprocessor = ChainedEncoder(
#         TensorEncoder(names=("time", "value", "index")),
#         DataFrameEncoder(
#             column_encoders={
#                 "value": IdentityEncoder(),
#                 tuple(ts.columns): FloatEncoder("float32"),
#             },
#             index_encoders=MinMaxScaler() @ DateTimeEncoder(unit="h"),
#         ),
#         TripletEncoder(sparse=True),
#         Standardizer(),
#     )
# else:
#     self.preprocessor = preprocessor
