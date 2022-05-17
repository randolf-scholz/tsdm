r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = ["MIMIC_DeBrouwer"]

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np
import torch
from pandas import DataFrame, Index, MultiIndex
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from tsdm.datasets import MIMIC_III
from tsdm.tasks.base import BaseTask
from tsdm.util.strings import repr_namedtuple


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)


class Batch(NamedTuple):
    r"""A single sample of the data."""

    T: Tensor  # B×N: the timestamps.
    X: Tensor  # B×N×D: the observations.
    Y: Tensor  # B×K×D: the target values.
    M: Tensor  # B×N: which t correspond to targets.

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)


@dataclass
class TaskDataset(torch.utils.data.Dataset):
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

    def __getitem__(self, key: int) -> Sample:
        t, x = self.tensors[key]
        observation_mask = t <= self.observation_time
        first_target = observation_mask.sum()
        target_mask = slice(first_target, first_target + self.prediction_steps)
        return Sample(
            key=key,
            inputs=Inputs(t[observation_mask], x[observation_mask], t[target_mask]),
            targets=x[target_mask],
            originals=(t, x),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


def my_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch."""
    t_list: list[Tensor] = []
    x_list: list[Tensor] = []
    m_list: list[Tensor] = []
    y_list: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        mask = torch.cat(
            (
                torch.zeros_like(t, dtype=torch.bool),
                torch.ones_like(t_target, dtype=torch.bool),
            )
        )
        x_padder = torch.full((t_target.shape[0], x.shape[-1]), fill_value=torch.nan)
        time = torch.cat((t, t_target))
        values = torch.cat((x, x_padder))
        idx = torch.argsort(time)
        t_list.append(time[idx])
        x_list.append(values[idx])
        m_list.append(mask[idx])
        y_list.append(sample.targets)

    T = pad_sequence(t_list, batch_first=True, padding_value=torch.nan).squeeze()
    X = pad_sequence(x_list, batch_first=True, padding_value=torch.nan).squeeze()
    Y = pad_sequence(y_list, batch_first=True, padding_value=torch.nan).squeeze()
    M = pad_sequence(m_list, batch_first=True, padding_value=False).squeeze()

    return Batch(T, X, Y, M)


class MIMIC_DeBrouwer(BaseTask):
    r"""Preprocessed subset of the MIMIC-III clinical dataset used by De Brouwer et al.

    Evaluation Protocol
    -------------------

    The subset of 96 variables that we use in our study are shown in Table 5. For each of those, we
    harmonize the units and drop the uncertain occurrences. We also remove outliers by discarding the
    measurements outside the 5 standard deviations interval. For models requiring binning of the time
    series, we map the measurements in 30-minute time bins, which gives 97 bins for 48 hours. When
    two observations fall in the same bin, they are either averaged or summed depending on the nature
    of the observation. Using the same taxonomy as in Table 5, lab measurements are averaged, while
    inputs, outputs, and prescriptions are summed.
    This gives a total of 3,082,224 unique measurements across all patients, or an average of 145
    measurements per patient over 48 hours.

    References
    ----------
    - | `GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series
        <https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html>`_
      | De Brouwer, Edward and Simm, Jaak and Arany, Adam and Moreau, Yves
      | `Advances in Neural Information Processing Systems 2019
        <https://proceedings.neurips.cc/paper/2019>`_
    """

    observation_time = 150
    prediction_steps = 3
    num_folds = 5
    seed = 432
    test_size = 0.1
    valid_size = 0.2

    def __init__(self):
        super().__init__()

        self.IDs = self.dataset.reset_index()["UNIQUE_ID"].unique()

    @cached_property
    def dataset(self) -> DataFrame:
        r"""Load the dataset."""
        return MIMIC_III()["observations"]

    @cached_property
    def folds(self) -> list[dict[str, Sequence[int]]]:
        r"""Create the folds."""
        num_folds = 5
        folds = []
        np.random.seed(self.seed)
        for _ in range(num_folds):
            train_idx, test_idx = train_test_split(self.IDs, test_size=self.test_size)
            train_idx, valid_idx = train_test_split(
                train_idx, test_size=self.valid_size
            )
            folds.append(
                {
                    "train": train_idx,
                    "valid": valid_idx,
                    "test": test_idx,
                }
            )

        return folds

    @cached_property
    def split_idx(self):
        r"""Create the split index."""
        fold_idx = Index(list(range(len(self.folds))), name="fold")
        splits = DataFrame(index=self.IDs, columns=fold_idx, dtype="string")

        for k in range(self.num_folds):
            for key, split in self.folds[k].items():
                mask = splits.index.isin(split)
                splits[k] = splits[k].where(
                    ~mask, key
                )  # where cond is false is replaces with key
        return splits

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

    def get_dataloader(
        self, key: tuple[int, str], /, **dataloader_kwargs: Any
    ) -> DataLoader:
        """Return the dataloader for the given key."""
        fold, partition = key
        fold_idx = self.folds[fold][partition]
        dataset = TaskDataset(
            [val for idx, val in self.tensors.items() if idx in fold_idx],
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
        )

        kwargs: dict[str, Any] = {"collate_fn": lambda *x: x} | dataloader_kwargs
        return DataLoader(dataset, **kwargs)
