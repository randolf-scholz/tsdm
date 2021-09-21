r"""Module Summary Line.

Module description
"""  # pylint: disable=line-too-long # noqa

from __future__ import annotations

import logging


import torch
from typing import Final, Literal
from pandas import Timestamp
from tsdm.util.dtypes import TimeStampLike
from tsdm.util.dataloaders import SliceSampler
from tsdm.datasets import Dataset, DATASETS
from tsdm.losses import Loss, LOSSES

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = []


class ETTh1_Informer:
    """

    Paper
    -----

    Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
    Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang
    https://arxiv.org/abs/2012.07436

    Evaluation Protocol
    -------------------


        ETT (Electricity Transformer Temperature)2: The ETT is a crucial indicator in
        the electric power long-term deployment. We collected 2-year data from two
        separated counties in China. To explore the granularity on the LSTF problem,
        we create separate datasets as {ETTh1, ETTh2}for 1-hour-level and ETTm1 for
        15-minute-level. Each data point consists of the target value ”oil temperature”
        and 6 power load features. The train/val/test is 12/4/4 months

        **Setup:** The input of each dataset is zero-mean normalized.

        For all methods, the input length of recurrent component is chosen from
        {24, 48, 96, 168, 336, 720} for the ETTh1, ETTh2, Weather and Electricity
        dataset, and chosen from {24, 48, 96, 192, 288, 672} for the ETTm dataset.

        The length of encoder’s input sequence and decoder’s start token is chosen from
        {24, 48, 96, 168, 336, 480, 720} for the ETTh1, ETTh2, Weather and ECL dataset,
         and {24, 48, 96, 192, 288, 480, 672}for the ETTm dataset.

        In the experiment, the decoder’s start token is a segment truncated from the
        encoder’s input sequence, so the length of decoder’s start token must be less
        than the length of encoder’s input.

    **Forcasting Horizon:** {1d, 2d, 7d, 14d, 30d, 40d}
    **Observation Horizon:**
    **Input_Length**: {24, 48, 96, 168, 336, 720}


    Test-Metric
    -----------

    - MSE: `⅟ₙ∑_{i=1}^n |y - ŷ|^2`
    - MAE `⅟ₙ∑_{i=1}^n |y - ŷ|`

    Results
    -------
    """
    dataset: Dataset
    test_metric: Loss
    observation_horizon: int
    forecasting_horizon: int

    def __init__(
        self,
        dataset: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"] = "ETTh1",
        forecasting_horizon: Literal[24,  48, 168, 336, 960] = 24,
        observation_horizon: Literal[24, 48, 96, 168, 336, 720] = 96,
        test_metric: Literal["MSE", "MAE"] = "MSE",
    ):
        self.dataset = DATASETS[dataset]
        self.test_metric = LOSSES[test_metric]
        self.forecasting_horizon = forecasting_horizon
        self.observation_horizon = observation_horizon

        self.train_dataset = self.dataset[:"2017-06-30"]              # inclusive range!
        self.valid_dataset = self.dataset["2017-07-01":"2017-10-31"]  # inclusive range!
        self.score_dataset = self.dataset["2017-11-01":"2018-02-28"]  # inclusive range!

    def test_loader(self, batch_size: int = 32):
        r"""Return a dataloader object with the given batch_size."""

