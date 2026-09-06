r"""Predicting transformer oil temperatures."""

__all__ = ["ETT_Zhou2021"]


from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import torch
from pandas import Series
from torch import Tensor, nn
from torch.utils.data import TensorDataset, default_collate

from tsdm.datasets import ETT
from tsdm.encoders import (
    DateTimeEncoder,
    Encoder,
    MinMaxScaler,
    StandardScaler,
)
from tsdm.encoders.pandas import FrameDTypeConverter, FrameEncoder
from tsdm.random.samplers import Sampler, SlidingWindowSampler
from tsdm.tasks.base import TimeSeriesTask
from tsdm.timeseries.pandas import TimeSeries

type SplitID = Literal["train", "trainval", "valid", "test"]

type Target = Literal["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

type DatasetID = Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


class ETT_Zhou2021(
    TimeSeriesTask[
        SplitID,
        Any,
        tuple[Tensor, ...],
        tuple[Tensor, ...],
    ]
):
    r"""Forecasting Oil Temperature on the Electrical-Transformer dataset.

    Paper
    -----

    - | Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
      | Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang
      | https://ojs.aaai.org/index.php/AAAI/article/view/17325

    Evaluation Protocol
    -------------------

    .. epigraph::

        ETT (Electricity Transformer Temperature)2: The ETT is a crucial indicator in
        the electric power long-term deployment. We collected 2-year data from two
        separated counties in China. To explore the granularity on the LSTF problem,
        we create separate dataset as {ETTh1, ETTh2} for 1-hour-level and ETTm1 for
        15-minute-level. Each data point consists of the target value ”oil temperature”
        and 6 power load features. The train/val/test is 12/4/4 months

        **Setup:** The input of each dataset is zero-mean normalized.

        For all methods, the input length of recurrent component is chosen from
        {24, 48, 96, 168, 336, 720} for the ETTh1, ETTh2, Weather and Electricity
        dataset, and chosen from {24, 48, 96, 192, 288, 672} for the ETTm dataset.

        The length of preprocessor’s input sequence and decoder’s start token is chosen from
        {24, 48, 96, 168, 336, 480, 720} for the ETTh1, ETTh2, Weather and ECL dataset,
        and {24, 48, 96, 192, 288, 480, 672}for the ETTm dataset.

        In the experiment, the decoder’s start token is a segment truncated from the
        preprocessor’s input sequence, so the length of decoder’s start token must be less
        than the length of preprocessor’s input.

        Appendix E
        [...]
        All the dataset are performed standardization such that the mean of variable
        is 0 and the standard deviation is 1.

    **Forecasting Horizon:** {1d, 2d, 7d, 14d, 30d, 40d}
    **Observation Horizon:**
    **Input_Length**: {24, 48, 96, 168, 336, 720}

    Test-Metric
    -----------

    - MSE: :math:`⅟ₙ∑ᵢ₌₁ⁿ |ŷ - y|ⁿ`
    - MAE: :math:`⅟ₙ∑ᵢ₌₁ⁿ |ŷ - y|`

    Results
    -------

    TODO: add results
    """

    accumulation_function: Callable[..., Tensor]
    r"""Accumulates residuals into loss - usually mean or sum."""

    train_batch_size: int = 32
    r"""Default batch size."""
    eval_batch_size: int = 128
    r"""Default batch size when evaluating."""

    # additional attributes
    dataset: TimeSeries  # type: ignore[reportIncompatibleVariableOverride]
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
        timeseries = ETT().tables[dataset_id]
        dataset = TimeSeries(dataset_id, timeseries=timeseries)
        super().__init__(
            dataset=dataset,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        )
        self.dataset_id = dataset_id
        self.target = target
        self.forecasting_horizon = forecasting_horizon
        self.observation_horizon = observation_horizon
        self.horizon = self.observation_horizon + self.forecasting_horizon
        self.frequency = dataset.timeindex[1] - dataset.timeindex[0]
        self.accumulation_function = nn.Identity()
        self.preprocessor = self._make_encoder()

    def _make_encoder(self) -> Encoder:
        r"""Create and fit the preprocessing encoder for the specified split."""
        encoder = (
            FrameDTypeConverter(float)
            >> StandardScaler(axis=-1)
            >> FrameEncoder({"date": DateTimeEncoder() >> MinMaxScaler()})
        )
        train_split = self.splits[self.get_train_split("train")]
        encoder.fit(train_split.timeseries)
        return encoder

    def make_folds(self, /) -> Mapping[SplitID, Series]:
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
        return cast(
            "Callable[[list[tuple[Tensor, ...]]], tuple[Tensor, ...]]", default_collate
        )

    def make_generator(self, key: SplitID, /) -> TensorDataset:
        r"""Create the encoded tensor dataset for the specified split."""
        encoded = self.encoders[key].encode(self.splits[key].timeseries).reset_index()
        tensor = torch.tensor(encoded.values, dtype=torch.float32)
        return TensorDataset(tensor)

    def make_sampler(self, key: SplitID, /) -> Sampler:
        r"""Create full-length sliding windows for the specified split."""
        return SlidingWindowSampler(
            self.splits[key].timeindex,
            horizons=self.horizon * self.frequency,
            stride=self.frequency,
            mode="index",
            drop_last=True,
            shuffle=self.is_train_split(key),
        )

    def make_test_metric(self, key: SplitID, /) -> Callable[[Tensor, Tensor], Tensor]:  # ruff: ignore[ARG002]
        r"""Return the evaluation metric."""
        return nn.MSELoss()
