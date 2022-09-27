r"""MIMIC-II clinical dataset."""

__all__ = [
    "MIMIC_IV_Bilos2021",
    "mimic_collate",
    "Sample",
    "Batch",
    "TaskDataset",
]

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import torch
from pandas import DataFrame, Index, MultiIndex
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch import nan as NAN
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tsdm.datasets import MIMIC_IV_Bilos2021 as MIMIC_IV_Dataset
from tsdm.encoders import FrameEncoder, MinMaxScaler, Standardizer
from tsdm.tasks.base import BaseTask
from tsdm.utils import is_partition
from tsdm.utils.strings import repr_namedtuple


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor

    def __repr__(self) -> str:
        r"""Return string representation."""
        return repr_namedtuple(self, recursive=False)


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor
    originals: tuple[Tensor, Tensor]

    def __repr__(self) -> str:
        r"""Return string representation."""
        return repr_namedtuple(self, recursive=False)


class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)


@dataclass
class TaskDataset(Dataset):
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


# @torch.jit.script  # seems to break things
def mimic_collate(batch: list[Sample]) -> Batch:
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

        # get whole time interval
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


class MIMIC_IV_Bilos2021(BaseTask):
    r"""Preprocessed subset of the MIMIC-III clinical dataset used by De Brouwer et al.

    Evaluation Protocol
    -------------------

    Filtering approach. Following De Brouwer et al. [16], we use clinical database MIMIC-III [35],
    pre-processed to contain 21250 patients’ time series, with 96 features. We also process newly released
    MIMIC-IV [25, 36] to obtain 17874 patients. The details are in Appendix D.2. The goal is to predict
    the next three measurements in the 12-hour interval after the observation window of 36 hours.

    Table 2 shows that our GRU flow model (Equation 5) mostly outperforms GRU-ODE [16]. Addition-
    ally, we show that the ordinary ResNet flow with 4 stacked transformations (Equation 2) performs
    worse. The reason might be because it is missing GRU flow properties, such as boundedness. Similarly,
    an ODE with a regular neural network does not outperform GRU-ODE [16]. Finally, we report
    that the model with GRU flow requires 60% less time to run one training epoch.

    References
    ----------
    - | `Neural Flows: Efficient Alternative to Neural ODEs
        <https://proceedings.neurips.cc/paper/2021/hash/b21f9f98829dea9a48fd8aaddc1f159d-Abstract.html>`_
      | Marin Biloš, Johanna Sommer, Syama Sundar Rangapuram, Tim Januschowski, Stephan Günnemann
      | `Advances in Neural Information Processing Systems 2021
        <https://proceedings.neurips.cc/paper/2021>`_
    """

    observation_time = 2160  # corresponds to 36 hours after admission (freq=1min)
    prediction_steps = 3
    num_folds = 5
    RANDOM_STATE = 0
    test_size = 0.15  # of total
    valid_size = 0.2  # of train split size, i.e. 0.85*0.2=0.17

    encoder: FrameEncoder[Standardizer, dict[Any, MinMaxScaler]]

    def __init__(self, normalize_time: bool = True):
        super().__init__()
        self.encoder = FrameEncoder(
            column_encoders=Standardizer(),
            index_encoders={"time_stamp": MinMaxScaler()},
        )
        self.normalize_time = normalize_time
        self.IDs = self.dataset.reset_index()["hadm_id"].unique()

    @cached_property
    def dataset(self) -> DataFrame:
        r"""Load the dataset."""
        ds = MIMIC_IV_Dataset()

        # Standardization is performed over full data slice, including test!
        # https://github.com/mbilos/neural-flows-experiments/blob/
        # bd19f7c92461e83521e268c1a235ef845a3dd963/nfe/experiments/gru_ode_bayes/lib/get_data.py#L50-L63

        # Standardize the x-values, min-max scale the t values.
        ts = ds.dataset
        self.encoder.fit(ts)
        ts = self.encoder.encode(ts)
        index_encoder = self.encoder.index_encoders["time_stamp"]
        self.observation_time /= index_encoder.param.xmax  # type: ignore[assignment]

        # drop values outside 5 sigma range
        ts = ts[(-5 < ts) & (ts < 5)]
        ts = ts.dropna(axis=1, how="all").copy()
        return ts

    @cached_property
    def folds(self) -> list[dict[str, Sequence[int]]]:
        r"""Create the folds."""
        num_folds = 5
        folds = []
        # https://github.com/edebrouwer/gru_ode_bayes/blob/aaff298c0fcc037c62050c14373ad868bffff7d2/data_preproc/Climate/generate_folds.py#L10-L14
        for _ in range(num_folds):
            train_idx, test_idx = train_test_split(
                self.IDs, test_size=self.test_size, random_state=self.RANDOM_STATE
            )
            train_idx, valid_idx = train_test_split(
                train_idx, test_size=self.valid_size, random_state=self.RANDOM_STATE
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
            t = torch.tensor(s.index.values, dtype=torch.float32)
            x = torch.tensor(s.values, dtype=torch.float32)
            tensors[_id] = (t, x)
        return tensors

    def get_dataloader(
        self, key: tuple[int, str], /, **dataloader_kwargs: Any
    ) -> DataLoader:
        r"""Return the dataloader for the given key."""
        fold, partition = key
        fold_idx = self.folds[fold][partition]
        dataset = TaskDataset(
            [val for idx, val in self.tensors.items() if idx in fold_idx],
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
        )
        kwargs: dict[str, Any] = {"collate_fn": lambda *x: x} | dataloader_kwargs
        return DataLoader(dataset, **kwargs)


# Remark: The following code is found in the repo:
#
# t_val = 2.16, and time is divided by 1000
# if self.validation:
#     assert val_options is not None, 'Validation set options should be fed'
#     self.df_before = self.df.loc[self.df['Time'] <= val_options['T_val']].copy()
#     self.df_after = self.df.loc[self.df['Time'] > val_options['T_val']].sort_values('Time').copy()
#     if val_options.get("T_stop"):
#         self.df_after = self.df_after.loc[self.df_after['Time'] < val_options['T_stop']].sort_values('Time').copy()
#     self.df_after = self.df_after.groupby('ID').head(val_options['max_val_samples']).copy()
#     self.df = self.df_before  # We remove observations after T_val
#     self.df_after.ID = self.df_after.ID.astype(np.int)
#     self.df_after.sort_values('Time', inplace=True)
