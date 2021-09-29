r"""TODO: Module Summary Line.

TODO: Module description
"""

from __future__ import annotations

__all__ = [
    # Classes
    "ETDatasetInformer",
]


import logging
from typing import Callable, Literal, Optional

import torch
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader

from tsdm.config import DEFAULT_DEVICE, DEFAULT_DTYPE
from tsdm.datasets import DATASETS, Dataset, SequenceDataset
from tsdm.encoders import ENCODERS, Encoder
from tsdm.losses import LOSSES, Loss
from tsdm.tasks.tasks import BaseTask
from tsdm.util.samplers import SequenceSampler

LOGGER = logging.getLogger(__name__)


class ETDatasetInformer(BaseTask):
    """Forecasting Oil Temperature on the Electrical-Transformer datasets.

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
    """

    dataset: Dataset
    splits: dict[str, DataFrame]
    r"""Available splits: train/valid/joint/test"""
    test_metric: type[Loss]
    observation_horizon: int
    forecasting_horizon: int
    accumulation_function: Callable[..., Tensor]
    target: str
    r"""One of "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"."""

    def __init__(
        self,
        dataset: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"] = "ETTh1",
        forecasting_horizon: Literal[24, 48, 168, 336, 960] = 24,
        observation_horizon: Literal[24, 48, 96, 168, 336, 720] = 96,
        test_metric: Literal["MSE", "MAE"] = "MSE",
        time_encoder: Encoder = "time2float",
        target: str = "OT",
        scale: bool = True,
    ):
        super().__init__()
        self.target = target
        self.dataset = DATASETS[dataset]
        self.test_metric = LOSSES[test_metric]
        self.time_encoder = ENCODERS[time_encoder]

        self.forecasting_horizon = forecasting_horizon
        self.observation_horizon = observation_horizon
        self.horizon = self.observation_horizon + self.forecasting_horizon
        self.accumulation_function = nn.Identity()  # type: ignore[assignment]
        # TODO: fix type problems

        self.splits: dict[str, DataFrame] = {
            "train": self.dataset.dataset["2016-07-01":"2017-06-30"],  # type: ignore
            "valid": self.dataset.dataset["2017-07-01":"2017-10-31"],  # type: ignore
            "joint": self.dataset.dataset["2016-07-01":"2017-10-31"],  # type: ignore
            "test": self.dataset.dataset["2017-11-01":"2018-02-28"],  # type: ignore
        }

        if scale:
            self.encoder = StandardScaler()
            self.encoder.fit(self.splits["train"])
            # note that this also transforms all splits as they are mere pointers to the slices.
            self.encoder.transform(self.dataset.dataset, copy=False)

    def get_dataloader(
        self,
        split: Literal["train", "valid", "joint", "test"],
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        shuffle: bool = True,
        drop_last: bool = False,
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
        drop_last: bool = False

        Returns
        -------
        DataLoader
        """
        device = DEFAULT_DEVICE if device is None else device
        dtype = DEFAULT_DTYPE if dtype is None else dtype

        if split == "test":
            assert not shuffle, "Don't shuffle when evaualting test-dataset!"

        ds = self.splits[split]
        _T = self.time_encoder(ds.index)
        _X = ds.drop(columns=self.target).values
        _Y = ds[self.target].values

        T = torch.tensor(_T, device=device, dtype=dtype)
        X = torch.tensor(_X, device=device, dtype=dtype)
        Y = torch.tensor(_Y, device=device, dtype=dtype)

        dataset = SequenceDataset([T, X, Y])
        sampler = SequenceSampler(dataset, seq_len=self.horizon, shuffle=shuffle)

        return DataLoader(
            dataset, sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )
