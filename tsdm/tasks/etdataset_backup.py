r"""TODO: Module Summary Line.

TODO: Module description
"""

from __future__ import annotations

__all__ = [
    # Classes
    "ETDatasetInformer",
]


import logging
from typing import Any, Callable, Literal, Optional

import torch
from pandas import DataFrame
from torch import Tensor, nn
from torch.utils.data import DataLoader

from tsdm.config import DEFAULT_DEVICE, DEFAULT_DTYPE
from tsdm.datasets import DATASETS, Dataset, SequenceDataset
from tsdm.encoders import ENCODERS, Encoder
from tsdm.losses import LOSSES, Loss
from tsdm.tasks.tasks import BaseTask
from tsdm.util import initialize_from
from tsdm.util.samplers import SequenceSampler

LOGGER = logging.getLogger(__name__)


class ETDatasetInformer(BaseTask):
    r"""Forecasting Oil Temperature on the Electrical-Transformer datasets.

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

        The length of preprocessor’s input sequence and decoder’s start token is chosen from
        {24, 48, 96, 168, 336, 480, 720} for the ETTh1, ETTh2, Weather and ECL dataset,
        and {24, 48, 96, 192, 288, 480, 672}for the ETTm dataset.

        In the experiment, the decoder’s start token is a segment truncated from the
        preprocessor’s input sequence, so the length of decoder’s start token must be less
        than the length of preprocessor’s input.

        Appendix E
        [...]
        All the datasets are performed standardization such that the mean of variable
        is 0 and the standard deviation is 1.

    **Forecasting Horizon:** {1d, 2d, 7d, 14d, 30d, 40d}
    **Observation Horizon:**
    **Input_Length**: {24, 48, 96, 168, 336, 720}

    Test-Metric
    -----------

    - MSE: `⅟ₙ∑_{i=1}^n |y - ŷ|^2`
    - MAE `⅟ₙ∑_{i=1}^n |y - ŷ|`

    Results
    -------

    TODO: add results
    """

    keys = ["train", "test", "valid", "joint", "trial"]
    r"""Available keys."""
    KEYS = Literal["train", "test", "valid", "joint", "trial", "whole"]
    r"""Type Hint for keys."""
    dataset: Dataset
    r"""The dataset."""
    splits: dict[KEYS, DataFrame]
    r"""Available splits: train/valid/joint/test."""
    test_metric: Loss  # Callable[..., Tensor] | nn.Module
    r"""The target metric."""
    accumulation_function: Callable[..., Tensor]
    r"""Accumulates residuals into loss - usually mean or sum."""

    train_batch_size: int = 32
    """Default batch size."""
    eval_batch_size: int = 128
    """Default batch size when evaluating."""

    # additional attributes
    preprocessor: Encoder
    r"""Task specific pre-processor."""
    preencoder: Encoder
    r"""Encoder for the observations."""
    observation_horizon: Literal[24, 48, 96, 168, 336, 720] = 96
    r"""The number of datapoints observed during prediction."""
    forecasting_horizon: Literal[24, 48, 168, 336, 960] = 24
    r"""The number of datapoints the model should forecast."""
    TARGET = Literal["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    r"""Type hint available targets."""
    target: TARGET = "OT"
    r"""One of "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"."""

    def __init__(
        self,
        dataset: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"] = "ETTh1",
        forecasting_horizon: Literal[24, 48, 168, 336, 960] = 24,
        observation_horizon: Literal[24, 48, 96, 168, 336, 720] = 96,
        target: TARGET = "OT",
        scale: bool = True,
        eval_batch_size: int = 128,
        train_batch_size: int = 32,
        test_metric: Literal["MSE", "MAE"] = "MSE",
        preprocessor: str = "StandardScaler",
        time_encoder: str = "time2float",
    ):
        super().__init__()
        self.target = target
        self.forecasting_horizon = forecasting_horizon
        self.observation_horizon = observation_horizon
        self.eval_batch_size = eval_batch_size
        self.train_batch_size = train_batch_size

        self.preprocessor = initialize_from(ENCODERS, __name__=preprocessor)
        self.time_encoder = initialize_from(ENCODERS, __name__=time_encoder)
        self.dataset = initialize_from(DATASETS, __name__=dataset)
        self.test_metric = initialize_from(LOSSES, __name__=test_metric)  # type: ignore[assignment]

        self.horizon = self.observation_horizon + self.forecasting_horizon
        self.accumulation_function = nn.Identity()  # type: ignore[assignment]

        # TODO: fix type problems
        self.splits: dict[str, DataFrame] = {
            "train": self.dataset.dataset.loc["2016-07-01":"2017-06-30"],  # type: ignore
            "valid": self.dataset.dataset.loc["2017-07-01":"2017-10-31"],  # type: ignore
            "joint": self.dataset.dataset.loc["2016-07-01":"2017-10-31"],  # type: ignore
            "trial": self.dataset.dataset.loc["2017-11-01":"2018-02-28"],  # type: ignore
            "whole": self.dataset.dataset,  # type: ignore
        }
        self.splits["test"] = self.splits["trial"]  # alias

        # Fit the Preprocessors
        self.preprocessor.fit(self.splits["train"])

    def get_dataloader(
        self,
        split: KEYS,
        batch_size: int = 1,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any
    ) -> DataLoader:
        r"""Return a DataLoader for the training-dataset with the given batch_size.

        Parameters
        ----------
        split: Literal["train", "valid", "test"]
            Dataset part from which to construct the DataLoader
        batch_size: int = 32
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
            defaults to cuda if cuda is available.
        shuffle: bool = True

        Returns
        -------
        DataLoader
        """
        device = DEFAULT_DEVICE if device is None else device
        dtype = DEFAULT_DTYPE if dtype is None else dtype

        if split == "test":
            assert not shuffle, "Don't shuffle when evaluating test-dataset!"
        if split == "test" and "drop_last" in kwargs:
            assert not kwargs["drop_last"], "Don't drop when evaluating test-dataset!"

        ds = self.splits[split]

        # Apply Encoders
        _T = self.time_encoder(ds.index)  # type: ignore[attr-defined]
        _V = self.preprocessor.transform(ds)

        _X = ds.drop(columns=self.target).values
        _Y = ds[self.target].values

        T = torch.tensor(_T, device=device, dtype=dtype)
        X = torch.tensor(_X, device=device, dtype=dtype)
        Y = torch.tensor(_Y, device=device, dtype=dtype)

        dataset = SequenceDataset([T, X, Y])
        sampler = SequenceSampler(dataset, seq_len=self.horizon, shuffle=shuffle)

        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, **kwargs)
