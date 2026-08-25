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
    "Inputs",
    "Sample",
    "MIMIC_IV_Bilos2021",
    "MIMIC_IV_SampleGenerator",
    "mimic_collate",
]

import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, cast
from warnings import deprecated

import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch import Tensor, nan as NAN, nn
from torch.nn.utils.rnn import pad_sequence

from tsdm.datatools import folds_as_frame, is_partition
from tsdm.encoders import FrameEncoder, MinMaxScaler
from tsdm.pprint import pprint_repr
from tsdm.random.samplers import RandomSampler, Sampler
from tsdm.tasks.base import TimeSeriesTask
from tsdm.timeseries.pandas import PandasTSC, mimic_iv_bilos2021


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


@dataclass(slots=True)
class MIMIC_IV_SampleGenerator:
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


@pprint_repr
class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.


@deprecated("Consider using tasks.utils.collate_timeseries instead.")
def mimic_collate(batch: list[Sample], /) -> Batch:
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


type SplitID = tuple[int, Literal["train", "valid", "test"]]


class MIMIC_IV_Bilos2021(TimeSeriesTask[SplitID, int, Sample, Batch]):
    r"""Preprocessed subset of the MIMIC-III clinical dataset used by De Brouwer et al."""

    dataset: PandasTSC[int]  # type: ignore
    preprocessor: FrameEncoder | None

    observation_time = 2160  # corresponds to 36 hours after admission (freq=1min)
    prediction_steps = 3
    num_folds = 5
    RANDOM_STATE = 0
    train_size = 0.70
    valid_size = 0.15
    test_size = 0.15

    def __init__(self, *, normalize_time: bool = True) -> None:
        warnings.warn(
            "This task is included for historical reasons, but it has several defects:",
            UserWarning,
            stacklevel=2,
        )

        self.normalize_time = normalize_time
        dataset = mimic_iv_bilos2021()
        timeseries = dataset.timeseries

        if normalize_time:
            self.preprocessor = FrameEncoder({"time_stamp": MinMaxScaler()})
            self.preprocessor.fit(timeseries)
            timeseries = self.preprocessor.encode(timeseries)
            index_encoder = cast("MinMaxScaler", self.preprocessor["time_stamp"])
            self.observation_time /= index_encoder.xmax
        else:
            self.preprocessor = None

        timeseries = timeseries.astype("float32")
        dataset = PandasTSC(  # pyright: ignore[reportIncompatibleVariableOverride]
            dataset.name,
            timeseries=timeseries,
            timeseries_metadata=dataset.timeseries_metadata,
            static_covariates=dataset.static_covariates,
            static_covariates_metadata=dataset.static_covariates_metadata,
            constants=dataset.constants,
            constants_metadata=dataset.constants_metadata,
        )
        self.IDs = dataset.metaindex
        super().__init__(dataset=dataset)

    def make_folds(self, /) -> DataFrame:
        r"""Create the folds."""
        folds: list[dict[str, Sequence[int]]] = []
        # NOTE: all folds are the same due to fixed random state.
        # see https://github.com/mbilos/neural-flows-experiments/blob/bd19f7c92461e83521e268c1a235ef845a3dd963/nfe/experiments/gru_ode_bayes/lib/get_data.py#L66-L67
        for _ in range(self.num_folds):
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

        return folds_as_frame(folds, index=self.IDs, sparse=True)

    def make_collate_fn(self, _key: SplitID, /) -> Callable[[list[Sample]], Batch]:
        r"""Return the collate function for the specified split."""
        return mimic_collate

    def make_generator(self, key: SplitID, /) -> MIMIC_IV_SampleGenerator:
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

        return MIMIC_IV_SampleGenerator(
            tensors,
            observation_time=self.observation_time,
            prediction_steps=self.prediction_steps,
        )

    def make_sampler(self, key: SplitID, /) -> Sampler[int]:
        r"""Return the sampler for the specified split."""
        generator = cast("MIMIC_IV_SampleGenerator", self.generators[key])
        return RandomSampler(
            range(len(generator)),
            shuffle=self.is_train_split(key),
        )

    def make_test_metric(
        self,
        _key: SplitID,
        /,
    ) -> Callable[[Tensor, Tensor], Tensor]:
        r"""Return the test metric."""
        return nn.MSELoss()
