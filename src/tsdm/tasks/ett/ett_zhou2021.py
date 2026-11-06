r"""Predicting transformer oil temperatures."""

__all__ = ["ETT_Zhou2021"]


from collections.abc import Callable
from functools import cached_property
from typing import Any, ClassVar, Literal

import torch
from pandas import Series
from torch import Tensor, nn
from torch.utils.data import TensorDataset, default_collate

from tsdm.encoders import (
    DateTimeEncoder,
    Encoder,
    MinMaxScaler,
    StandardScaler,
)
from tsdm.encoders.pandas import FrameDTypeConverter, FrameEncoder
from tsdm.random.samplers import Sampler, SlidingWindowSampler
from tsdm.tasks.base import TimeSeriesTask
from tsdm.timeseries.pandas import TimeSeries, etth1, etth2, ettm1, ettm2

type SplitID = Literal["train", "trainval", "valid", "test"]

type Target = Literal["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

type DatasetID = Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


class ETT_Zhou2021(
    TimeSeriesTask[
        SplitID,
        TimeSeries,
        Any,
        tuple[Tensor, ...],
        tuple[Tensor, ...],
    ]
):
    r"""Forecast oil temperature on the Electricity Transformer Temperature dataset.

    References:
        - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting <br>
          Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang <br>
          <https://ojs.aaai.org/index.php/AAAI/article/view/17325>

    ### Evaluation Protocol

    The dataset contains oil temperature and six power-load features. `ETTh1` and
    `ETTh2` are sampled hourly; `ETTm1` and `ETTm2` are sampled every 15 minutes.
    The training, validation, and test partitions span 12, 4, and 4 months,
    respectively. The shared preprocessing encoder is fitted on the training partition.

    Observation and forecasting horizons are specified as numbers of time steps,
    so their durations depend on the dataset's sampling frequency. The default
    observation horizon is 96 steps and the default forecasting horizon is 24 steps.

    The paper standardizes each variable to mean zero and standard deviation one.
    It considers recurrent input lengths of 24, 48, 96, 168, 336, and 720 steps for
    the hourly datasets, and 24, 48, 96, 192, 288, and 672 steps for the minute datasets.
    Decoder start tokens are truncated from the encoder input and must be shorter
    than that input.

    The paper considers encoder input and decoder start-token lengths of 24, 48,
    96, 168, 336, 480, and 720 steps for the hourly datasets, and 24, 48, 96, 192,
    288, 480, and 672 steps for the minute datasets.

    ### Test Metrics

    For $n$ scalar predictions $ŷᵢ$ and targets $yᵢ$, the paper reports:

    - Mean squared error: $\frac{1}{n} ∑ᵢ₌₁ⁿ |ŷᵢ - yᵢ|²$.
    - Mean absolute error: $\frac{1}{n} ∑ᵢ₌₁ⁿ |ŷᵢ - yᵢ|$.

    This implementation uses `torch.nn.MSELoss` as its error function.
    """

    accumulation_function: Callable[..., Tensor]
    r"""Accumulates residuals into loss - usually mean or sum."""

    DEFAULT_TRAIN_BATCH_SIZE: ClassVar[int] = 32
    r"""Default batch size."""
    DEFAULT_EVAL_BATCH_SIZE: ClassVar[int] = 128
    r"""Default batch size when evaluating."""

    # additional attributes
    dataset: TimeSeries
    preprocessor: Encoder
    r"""Encoder for the observations."""
    observation_horizon: Literal[24, 48, 96, 168, 336, 720] = 96
    r"""The number of datapoints observed during prediction."""
    forecasting_horizon: Literal[24, 48, 168, 336, 960] = 24
    r"""The number of datapoints the model should forecast."""
    target: Target = "OT"
    r"""One of "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"."""
    dataset_id: DatasetID

    def __init__(
        self,
        dataset_id: DatasetID,
        *,
        forecasting_horizon: Literal[24, 48, 168, 336, 960] = 24,
        observation_horizon: Literal[24, 48, 96, 168, 336, 720] = 96,
        target: Target = "OT",
        eval_batch_size: int = 128,
        train_batch_size: int = 32,
    ) -> None:
        dataset = {
            "ETTh1": etth1,
            "ETTh2": etth2,
            "ETTm1": ettm1,
            "ETTm2": ettm2,
        }[dataset_id]()
        self.dataset_id = dataset_id
        self.target = target
        self.forecasting_horizon = forecasting_horizon
        self.observation_horizon = observation_horizon
        self.horizon = self.observation_horizon + self.forecasting_horizon
        self.frequency = dataset.timeindex.iloc[1] - dataset.timeindex.iloc[0]
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.accumulation_function = nn.Identity()
        super().__init__(dataset=dataset, error_fn=nn.MSELoss())
        self.preprocessor = self._make_encoder()

    @property
    def split_keys(self) -> tuple[SplitID, ...]:
        return tuple(self.split_masks)

    def is_train_split(self, key: SplitID) -> bool:
        return key in {"train", "trainval"}

    def make_partition(self, key: SplitID, /) -> TimeSeries:
        return self.dataset[self.split_masks[key].to_list()]

    def _make_encoder(self) -> Encoder:
        r"""Fit the shared preprocessing encoder on the training partition."""
        encoder = (
            FrameDTypeConverter(float)
            >> StandardScaler(axis=-1)
            >> FrameEncoder({"date": DateTimeEncoder() >> MinMaxScaler()})
        )
        train_split = self.partitions["train"]
        encoder.fit(train_split.timeseries)
        return encoder

    @cached_property
    def split_masks(self) -> dict[SplitID, Series]:
        r"""Create timestamp masks for the prescribed ETT partitions."""
        timeseries = self.dataset.timeseries
        index = timeseries.index

        def mask(start: str, end: str, /) -> Series:
            return Series(
                index.isin(timeseries.loc[start:end].index),
                index=index,
            )

        return {
            "train": mask("2016-07-01", "2017-06-30"),
            "valid": mask("2017-07-01", "2017-10-31"),
            "trainval": mask("2016-07-01", "2017-10-31"),
            "test": mask("2017-11-01", "2018-02-28"),
        }

    def make_collate_fn(
        self, _key: SplitID, /
    ) -> Callable[[list[tuple[Tensor, ...]]], tuple[Tensor, ...]]:
        r"""Return PyTorch's default collate function."""
        return lambda samples: tuple(default_collate(samples))

    def make_generator(self, key: SplitID, /) -> TensorDataset:
        r"""Create the encoded tensor dataset for the specified split."""
        encoded = self.preprocessor.encode(
            self.partitions[key].timeseries
        ).reset_index()
        tensor = torch.tensor(encoded.values, dtype=torch.float32)
        return TensorDataset(tensor)

    def make_sampler(self, key: SplitID, /) -> Sampler:
        r"""Create full-length sliding windows for the specified split."""
        return SlidingWindowSampler(
            self.partitions[key].timeindex,
            horizons=self.horizon * self.frequency,
            stride=self.frequency,
            mode="index",
            drop_last=True,
            shuffle=self.is_train_split(key),
        )
