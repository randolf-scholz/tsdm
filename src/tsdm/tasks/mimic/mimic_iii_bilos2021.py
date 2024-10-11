r"""MIMIC-III Forecasting Task as described by Bilos et al. (2021) [1]_.

Evaluation Protocol
-------------------

.. epigraph::

    Filtering approach. Following De Brouwer et al. [16], we use clinical database MIMIC-III [35],
    pre-processed to contain 21250 patients’ time series, with 96 features. We also process newly released
    MIMIC-IV [25, 36] to obtain 17874 patients. The details are in Appendix D.2. The goal is to predict
    the next three measurements in the 12-hour interval after the observation window of 36 hours.

    Table 2 shows that our GRU flow model (Equation 5) mostly outperforms GRU-ODE [16]. Additionally,
    we show that the ordinary ResNet flow with 4 stacked transformations (Equation 2) performs
    worse. The reason might be because it is missing GRU flow properties, such as boundedness. Similarly,
    an ODE with a regular neural network does not outperform GRU-ODE [16]. Finally, we report
    that the model with GRU flow requires 60% less time to run one training epoch.

Notes:
    - Authors code is available at [2]_.
    - The authors use a 70/15/15 split for train/valid/test. This is not mentioned in the paper but can
      be seen in the code. Moreover, the authors (accidentally?) use the same random seed (0) for each fold [3]_.
      This explains the low reported values for the standard deviation::

            train_idx, eval_idx = train_test_split(
                full_data.index.unique(),
                test_size=0.3,
                random_state=0
            )
            val_idx, test_idx = train_test_split(
                full_data.loc[eval_idx].index.unique(),
                test_size=0.5,
                random_state=0
            )
    - on MIMIC-IV, the authors remove 5-sigma outliers, cf. [4]_

References:
    .. [1] | `Neural Flows: Efficient Alternative to Neural ODEs <https://proceedings.neurips.cc/paper/2021/hash/b21f9f98829dea9a48fd8aaddc1f159d-Abstract.html>`_
           | Marin Biloš, Johanna Sommer, Syama Sundar Rangapuram, Tim Januschowski, Stephan Günnemann.
            `Advances in Neural Information Processing Systems 2021 <https://proceedings.neurips.cc/paper/2021>`_
    .. [2] https://github.com/mbilos/neural-flows-experiments/
    .. [3] https://github.com/mbilos/neural-flows-experiments/blob/bd19f7c92461e83521e268c1a235ef845a3dd963/nfe/experiments/gru_ode_bayes/lib/get_data.py#L66-L67
    .. [4] https://github.com/mbilos/neural-flows-experiments/blob/bd19f7c92461e83521e268c1a235ef845a3dd963/nfe/experiments/gru_ode_bayes/lib/get_data.py#L55-L63
"""

__all__ = [
    "Batch",
    "Sample",
    "Inputs",
    "MIMIC_III_Bilos2021",
    "MIMIC_III_SampleGenerator",
    "mimic_iii_collate",
]

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import torch
from pandas import DataFrame, Index, MultiIndex
from sklearn.model_selection import train_test_split
from torch import Tensor, nan as NAN, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from typing_extensions import deprecated

from tsdm.data import is_partition
from tsdm.datasets import MIMIC_III_Bilos2021 as MIMIC_III_Dataset
from tsdm.tasks._deprecated import OldBaseTask
from tsdm.utils.decorators import pprint_repr


@pprint_repr
class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor


@pprint_repr
class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]


@pprint_repr
class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.


@pprint_repr
@dataclass
class MIMIC_III_SampleGenerator(Dataset):
    r"""Wrapper for creating samples of the dataset."""

    tensors: list[tuple[Tensor, Tensor]]
    observation_time: float
    prediction_steps: int

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return len(self.tensors)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        r"""Return an iterator over the dataset."""
        return iter(self.tensors)

    def __getitem__(self, key: int, /) -> Sample:
        t, x = self.tensors[key]
        observations = t <= self.observation_time
        first_target = observations.sum()
        sample_mask = slice(0, first_target)
        target_mask = slice(first_target, first_target + self.prediction_steps)
        return Sample(
            key=key,
            inputs=Inputs(t[sample_mask], x[sample_mask], t[target_mask]),
            targets=x[target_mask],
            originals=(t, x),
        )


@deprecated("Consider using tasks.utils.collate_timeseries instead.")
def mimic_iii_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get the whole time interval
        time = torch.cat((t, t_target))
        sorted_idx = torch.argsort(time)

        # pad the x-values
        x_padding = torch.full(
            (t_target.shape[0], x.shape[-1]), fill_value=NAN, device=x.device
        )
        values = torch.cat((x, x_padding))

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_pad = torch.zeros_like(x, dtype=torch.bool)
        mask_x = torch.cat((mask_pad, mask_y))

        x_vals.append(values[sorted_idx])
        x_time.append(time[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)

    return Batch(
        x_time=pad_sequence(x_time, batch_first=True).squeeze(),
        x_vals=pad_sequence(x_vals, batch_first=True, padding_value=NAN).squeeze(),
        x_mask=pad_sequence(x_mask, batch_first=True).squeeze(),
        y_time=pad_sequence(y_time, batch_first=True).squeeze(),
        y_vals=pad_sequence(y_vals, batch_first=True, padding_value=NAN).squeeze(),
        y_mask=pad_sequence(y_mask, batch_first=True).squeeze(),
    )


class MIMIC_III_Bilos2021(OldBaseTask):
    r"""MIMIC-III Forecasting Task as described by Bilos et al. (2021)."""

    observation_time = 75  # corresponds to 36 hours after admission (freq=30min)
    prediction_steps = 3
    num_folds = 5
    seed = 432
    RANDOM_STATE = 0
    train_size = 0.70
    valid_size = 0.15
    test_size = 0.15

    def __init__(self, *, normalize_time: bool = True) -> None:
        super().__init__()
        if self.train_size + self.test_size + self.valid_size != 1:
            raise ValueError("The train/valid/test split must sum to 1.")
        self.normalize_time = normalize_time
        self.IDs = self.dataset.reset_index()["UNIQUE_ID"].unique()

    @cached_property
    def dataset(self) -> DataFrame:
        r"""Load the dataset."""
        ts = MIMIC_III_Dataset().table
        # SEE: https://github.com/edebrouwer/gru_ode_bayes/blob/aaff298c0fcc037c62050c14373ad868bffff7d2/data_preproc/Climate/generate_folds.py#L10-L14
        if self.normalize_time:
            ts = ts.reset_index()
            t_max = ts["TIME_STAMP"].max()
            self.observation_time /= t_max
            ts["TIME_STAMP"] /= t_max
            ts = ts.set_index(["UNIQUE_ID", "TIME_STAMP"])

        # NOTE: only numpy float types supported by torch
        ts = ts.dropna(axis=1, how="all").copy().astype("float32")
        return ts

    @cached_property
    def folds(self) -> list[dict[str, Sequence[int]]]:
        r"""Create the folds."""
        num_folds = 5
        folds = []
        # NOTE: all folds are the same due to fixed random state.
        # SEE: https://github.com/mbilos/neural-flows-experiments/blob/bd19f7c92461e83521e268c1a235ef845a3dd963/nfe/experiments/gru_ode_bayes/lib/get_data.py#L66-L67
        for _ in range(num_folds):
            train_idx, test_idx = train_test_split(
                self.IDs,
                test_size=self.test_size
                / (self.train_size + self.valid_size + self.test_size),
                random_state=self.RANDOM_STATE,
            )
            train_idx, valid_idx = train_test_split(
                train_idx,
                test_size=self.valid_size / (self.train_size + self.valid_size),
                random_state=self.RANDOM_STATE,
            )
            fold = {
                "train": train_idx,
                "valid": valid_idx,
                "test": test_idx,
            }
            if not is_partition(fold.values(), union=self.IDs):
                raise ValueError("Invalid partitions!")
            folds.append(fold)

        return folds

    @cached_property
    def split_idx(self) -> DataFrame:
        r"""Create the split index."""
        fold_idx = Index(list(range(len(self.folds))), name="fold")
        splits = DataFrame(index=self.IDs, columns=fold_idx, dtype="string")

        for k in range(self.num_folds):
            for key, split in self.folds[k].items():
                mask = splits.index.isin(split)
                splits[k] = splits[k].where(
                    ~mask, key
                )  # where cond is `False`, the value is replaced with 'key'.
        return splits

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
    def test_metric(self) -> Callable[[Tensor, Tensor], Tensor]:
        r"""The test metric."""
        return nn.MSELoss()

    @cached_property
    def splits(self) -> Mapping:
        r"""Create the splits."""
        splits = {}
        for key in self.index:
            mask = self.split_idx_sparse[key]
            ids = self.split_idx_sparse.index[mask]
            splits[key] = self.dataset.loc[ids]
        return splits

    @cached_property
    def index(self) -> MultiIndex:
        r"""Create the index."""
        return self.split_idx_sparse.columns

    @cached_property
    def tensors(self) -> Mapping:
        r"""Tensor dictionary."""
        tensors = {}
        for _id in self.IDs:
            s = self.dataset.loc[_id]
            t = torch.tensor(s.index.values, dtype=torch.float32)
            x = torch.tensor(s.values, dtype=torch.float32)
            tensors[_id] = (t, x)
        return tensors

    def make_dataloader(
        self, key: tuple[int, str], /, **dataloader_kwargs: Any
    ) -> DataLoader:
        r"""Return the dataloader for the given key."""
        fold, partition = key
        fold_idx = self.folds[fold][partition]
        dataset = MIMIC_III_SampleGenerator(
            [val for idx, val in self.tensors.items() if idx in fold_idx],
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
        )
        kwargs: dict[str, Any] = {"collate_fn": lambda x: x} | dataloader_kwargs
        return DataLoader(dataset, **kwargs)
