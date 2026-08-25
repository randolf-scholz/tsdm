r"""USHCN Forecasting Task as described by De Brouwer et al. (2019) [1]_.

Evaluation Protocol
-------------------

.. epigraph::

    5.3 Climate forecast

    From short-term weather forecast to long-range prediction or assessment of systemic
    changes, such as global warming, climatic data has always been a popular application for
    time-series analysis. This data is often considered to be regularly sampled over long
    periods of time, which facilitates their statistical analysis. Yet, this assumption does
    not usually hold in practice. Missing data are a problem that is repeatedly encountered in
    climate research because of, among others, measurement errors, sensor failure, or faulty
    data acquisition. The actual data is then sporadic and researchers usually resort to
    imputation before statistical analysis (Junninen et al., 2004; Schneider, 2001).

    We use the publicly available United States Historical Climatology Network (USHCN) daily
    data set (Menne et al.), which contains measurements of 5 climate variables
    (daily temperatures, precipitation, and snow) over 150 years for 1,218 meteorological
    stations scattered over the United States. We selected a subset of 1,114 stations and an
    observation window of 4 years (between 1996 and 2000). To make the time series sporadic, we
    subsample the data such that each station has an average of around 60 observations over
    those 4 years. Appendix L contains additional details regarding this procedure.
    The task is then to predict the next 3 measurements after the first 3 years of observation.

Notes:
    - Authors code is available at [2]_.
    - The author's code is missing the pre-processing script to create the folds for MIMIC-III. There is an open issue:
      https://github.com/edebrouwer/gru_ode_bayes/issues/15, but the authors are not responding.
      We assume the MIMIC-III pre-processing script is similar/the same as the one for the Climate dataset.

    - The train/valid/test split is 62/18/10. The authors write:

        We report the performance using 5-fold cross-validation. Hyperparameters (dropout and weight decay) are chosen
        using an inner holdout validation set (20%), and performance is assessed on a left-out test set (10%).

      Here, 20% validation refers to 20% of the non-test data, i.e. 20% of 90% = 18% of the total data. [3]_
    - The random seed is fixed to 432 at the start of the splitting process. [3]_

References:
    .. [1] | GRU-ODE-Bayes: Continuous Modeling of Sporadically﹣Observed Time Series
           | <https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html>_
           | De Brouwer, Edward and Simm, Jaak and Arany, Adam and Moreau, Yves.
           | `Advances in Neural Information Processing Systems 2019 <https://proceedings.neurips.cc/paper/2019>`_
    .. [2] https://github.com/edebrouwer/gru_ode_bayes
    .. [3] https://github.com/edebrouwer/gru_ode_bayes/blob/master/data_preproc/Climate/generate_folds.py
"""

__all__ = [
    "Batch",
    "Inputs",
    "Sample",
    "USHCN_DeBrouwer2019",
    "USHCN_SampleGenerator",
    "ushcn_collate",
]

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple, cast
from warnings import deprecated

import numpy as np
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch import Tensor, nan as NAN, nn
from torch.nn.utils.rnn import pad_sequence

from tsdm.datatools import folds_as_frame, is_partition
from tsdm.pprint import pprint_repr
from tsdm.random.samplers import RandomSampler, Sampler
from tsdm.tasks.base import TimeSeriesTask
from tsdm.timeseries.pandas import PandasTSC, ushcn_de_brouwer2019


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
@dataclass(slots=True)
class USHCN_SampleGenerator:
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
def ushcn_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: `t, x, t_target → T, X where X[t_target:] = NAN`.
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


type SplitID = tuple[int, Literal["train", "valid", "test"]]


class USHCN_DeBrouwer2019(TimeSeriesTask[SplitID, int, Sample, Batch]):
    r"""USHCN Forecasting Task as described by De Brouwer et al. (2019)."""

    dataset: PandasTSC  # type: ignore

    observation_time = 150
    prediction_steps = 3
    num_folds = 5
    seed = 432
    test_size = 0.1  # of total
    valid_size = 0.2  # of train, i.e. 0.9*0.2 = 0.18

    def __init__(self, *, normalize_time: bool = True) -> None:
        self.normalize_time = normalize_time
        dataset = cast("PandasTSC[int]", ushcn_de_brouwer2019())
        timeseries = dataset.timeseries

        if normalize_time:
            timeseries = timeseries.reset_index()
            t_max = timeseries["Time"].max()
            self.observation_time /= t_max
            timeseries["Time"] /= t_max
            timeseries = timeseries.set_index(["ID", "Time"])

        # NOTE: only numpy float types supported by torch
        timeseries = timeseries.dropna(axis=1, how="all").copy().astype("float32")
        self.dataset = replace(dataset, timeseries=timeseries)
        self.IDs = self.dataset.metaindex
        super().__init__(dataset=self.dataset)

    def make_folds(self, /) -> DataFrame:
        r"""Create the folds."""
        folds: list[dict[str, Sequence[int]]] = []
        rng = np.random.RandomState(self.seed)

        # https://github.com/edebrouwer/gru_ode_bayes/blob/aaff298c0fcc037c62050c14373ad868bffff7d2/data_preproc/Climate/generate_folds.py#L10-L14
        for _ in range(self.num_folds):
            train_idx, test_idx = train_test_split(
                self.IDs,
                test_size=self.test_size,
                random_state=rng,
            )
            train_idx, valid_idx = train_test_split(
                train_idx,
                test_size=self.valid_size,
                random_state=rng,
            )
            fold = {
                "train": train_idx,
                "valid": valid_idx,
                "test": test_idx,
            }
            if not is_partition(fold.values(), union=self.IDs):
                raise ValueError("Invalid partition!")
            folds.append(fold)

        return folds_as_frame(folds, index=self.IDs, sparse=True)

    def make_collate_fn(self, key: SplitID, /) -> Callable[[list[Sample]], Batch]:  # ruff: ignore[ARG002]
        r"""Return the collate function for the specified split."""
        return cast("Callable[[list[Sample]], Batch]", ushcn_collate)

    def make_generator(self, key: SplitID, /) -> USHCN_SampleGenerator:
        r"""Return the sample generator for the specified split."""
        tensors: list[tuple[Tensor, Tensor]] = []
        for identifier in self.splits[key]:
            timeseries = self.dataset[identifier]
            tensors.append(
                (
                    torch.tensor(timeseries.timeindex.values, dtype=torch.float32),
                    torch.tensor(timeseries.timeseries.values, dtype=torch.float32),
                )
            )

        return USHCN_SampleGenerator(
            tensors,
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
        )

    def make_sampler(self, key: SplitID, /) -> Sampler[int]:
        r"""Return the sampler for the specified split."""
        generator = cast("USHCN_SampleGenerator", self.generators[key])
        return RandomSampler(
            range(len(generator)),
            shuffle=self.is_train_split(key),
        )

    def make_test_metric(self, key: SplitID, /) -> Callable[[Tensor, Tensor], Tensor]:  # ruff: ignore[ARG002]
        r"""Return the test metric."""
        return nn.MSELoss()
