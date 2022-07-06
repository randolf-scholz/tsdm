r"""MIMIC-II clinical dataset."""

__all__ = ["MIMIC_DeBrouwer", "mimic_collate"]

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
from tsdm.util import is_partition
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

    TX: Tensor  # B×N:   the input timestamps.
    VX: Tensor  # B×N×D: the input values.
    MX: Tensor  # B×N×D: the input mask.

    TY: Tensor  # B×K:   the target timestamps.
    VY: Tensor  # B×K×D: the target values.
    MY: Tensor  # B×K×D: teh target mask.

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


# @torch.jit.script  <--seems to be slower!?
def mimic_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch."""
    vx_list: list[Tensor] = []
    vy_list: list[Tensor] = []
    tx_list: list[Tensor] = []
    ty_list: list[Tensor] = []
    mx_list: list[Tensor] = []
    my_list: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        time = torch.cat((t, t_target))
        sorted_idx = torch.argsort(time)

        mask_y = y.isfinite()

        mask_x = torch.cat(
            (
                torch.zeros_like(x, dtype=torch.bool),
                mask_y,
            )
        )

        x_padder = torch.full((t_target.shape[0], x.shape[-1]), fill_value=torch.nan)
        values = torch.cat((x, x_padder))

        vx_list.append(values[sorted_idx])
        vy_list.append(y)
        tx_list.append(time[sorted_idx])
        ty_list.append(t_target)
        mx_list.append(mask_x[sorted_idx])
        my_list.append(mask_y)

    VX = pad_sequence(vx_list, batch_first=True, padding_value=torch.nan).squeeze()
    VY = pad_sequence(vy_list, batch_first=True, padding_value=torch.nan).squeeze()
    TX = pad_sequence(tx_list, batch_first=True).squeeze()
    TY = pad_sequence(ty_list, batch_first=True).squeeze()
    MX = pad_sequence(mx_list, batch_first=True).squeeze()
    MY = pad_sequence(my_list, batch_first=True).squeeze()

    return Batch(TX, VX, MX, TY, VY, MY)


class MIMIC_DeBrouwer(BaseTask):
    r"""Preprocessed subset of the MIMIC-III clinical dataset used by De Brouwer et al.

    Evaluation Protocol
    -------------------

    We use the publicly available MIMIC-III clinical database (Johnson et al., 2016), which contains
    EHR for more than 60,000 critical care patients. We select a subset of 21,250 patients with sufficient
    observations and extract 96 different longitudinal real-valued measurements over a period of 48 hours
    after patient admission. We refer the reader to Appendix K for further details on the cohort selection.
    We focus on the predictions of the next 3 measurements after a 36-hour observation window.

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

    observation_time = 75  # corresponds to 36 hours after admission
    prediction_steps = 3
    num_folds = 5
    seed = 432
    test_size = 0.1
    valid_size = 0.2
    device: torch.device

    def __init__(self, normalize_time: bool = False, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.normalize_time = normalize_time
        self.IDs = self.dataset.reset_index()["UNIQUE_ID"].unique()

    @cached_property
    def dataset(self) -> DataFrame:
        r"""Load the dataset."""
        ds = MIMIC_III()["observations"]

        if self.normalize_time:
            ds = ds.reset_index()
            t_max = ds["TIME_STAMP"].max()
            self.observation_time /= t_max
            ds["TIME_STAMP"] /= t_max
            ds = ds.set_index(["UNIQUE_ID", "TIME_STAMP"])
        return ds

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
            fold = {
                "train": train_idx,
                "valid": valid_idx,
                "test": test_idx,
            }
            assert is_partition(fold.values(), union=self.IDs)
            folds.append(fold)

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
            t = torch.tensor(s.index.values, dtype=torch.float32, device=self.device)
            x = torch.tensor(s.values, dtype=torch.float32, device=self.device)
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
