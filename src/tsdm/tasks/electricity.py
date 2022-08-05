r"""Tasks associated with the Electricity dataset."""


__all__ = [
    # CLASSES
    "ElectricityDeepState",
    "ElectricityDeepAR",
    "ElectricityTFT",
    "ElectricityTRMF",
    "ElectricityELBMBTTF",
]

from functools import cached_property
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from tsdm.datasets import Electricity
from tsdm.encoders.modular import BaseEncoder, Standardizer
from tsdm.random.samplers import SequenceSampler
from tsdm.tasks.base import BaseTask


class ElectricityDeepState:
    r"""Experiments as performed by the "DeepState" paper.

    Paper
    -----
    - | Deep State Space Models for Time Series Forecasting
      | Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang,
        Tim Januschowski
      | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
      | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html

    Evaluation Protocol
    -------------------

        We train each method on all time series of these dataset but vary the size of the training
        range Tᵢ∈ {14, 21, 28} days. We evaluate all the methods on the next τ = 7 days after the
        forecast start time using the standard p50 and p90- quantile losses.

    Test-Metric
    -----------

    Results
    -------
    Observation horizons: [14, 21, 28] days
    Forecast    horizons: 7 days
    Split:

    NBEATS claims a key at 2014-09-01 is used. But this seems wrong.
    The date 2014-09-01 only ever occurs in Appendix A5, Figure 4&5 which show an example plot.
    """


class ElectricityDeepAR:
    r"""Experiments as performed by the "DeepAR" paper.

    Paper
    -----

    - | DeepAR: Probabilistic forecasting with autoregressive recurrent networks
      | http://www.sciencedirect.com/science/article/pii/S0169207019301888>

    Evaluation Protocol
    -------------------

        For electricity we train with data between 2014-01-01 and 2014-09-01, for traffic we train
        all the data available before 2008-06-15. The results for electricity and traffic are
        computed using rolling window predictions done after the last point seen in training as
        described in [23]. We do not retrain our model for each window, but use a single model
        trained on the data before the first prediction window.

    Test-Metric
    -----------

    Results
    -------
    """


class ElectricityTRMF:
    r"""Experiments as performed by the "TRMF" paper.

    Paper
    -----

    - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
      | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html

    Evaluation Protocol
    -------------------

        5.1 Forecasting
        [...]
        For electricity and traffic, we consider the 24-hour ahead forecasting task and use last
        seven days as the test periods.

        A.1 Datasets and Evaluation Criteria
        [...]
        electricity 3 : the electricity usage in kW recorded every 15 minutes, for n = 370 clients.
        We convert the data to reflect hourly consumption, by aggregating blocks of 4 columns,
        to obtain T = 26, 304. Teh coefficient of variation for electricity is 6.0341.

    Test-Metric
    -----------

    **Normalized deviation (ND)**

    .. math::
        𝖭𝖣(Y, Ŷ) = \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i,t) ∈ Ω_\text{test}}|Ŷ_{it}-Y_{it}|\Big)
        \Big/ \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i,t) ∈ Ω_\text{test}}|Y_{it}|\Big)

    **Normalized RMSE (NRMSE)**

    .. math::
        𝖭𝖱𝖬𝖲𝖤(Y, Ŷ) = \sqrt{\frac{1}{|Ω_\text{test}|}∑_{(i,t) ∈ Ω_\text{test}}|Ŷ_{it}-Y_{it}|^2}
        \Big/ \Big(\frac{1}{|Ω_\text{test}|} ∑_{(i,t) ∈ Ω_\text{test}}|Y_{it}|\Big)

    Results
    -------
    +-------+-------+-------------+-------------+---------------+
    | Model | TRMF  | N-BEATS (G) | N-BEATS (I) | N-BEATS (I+G) |
    +=======+=======+=============+=============+===============+
    | ND    | 0.255 | 0.171       | 0.185       | 0.111         |
    +-------+-------+-------------+-------------+---------------+
    """


class ElectricityTFT(BaseTask):
    r"""Experiments as performed by the "TFT" paper.

    Note that there is an issue: in the pipe-line, the hourly aggregation is done via mean,
    whereas in the TRMF paper, the hourly aggregation is done via sum.

    > We convert the data to reflect hourly consumption, by aggregating blocks of 4 columns,

    Issues:
    - They report in the paper: 90% train, 10% validation. However, this is wrong.
      They split the array not on the % of samples, but instead tehy use the first 218 days
      as train and the following 23 days ("2014-08-08" ≤ t < "2014-09-01" ) as validation,
      leading to a split of 90.12% train and 9.88% validation.
    - preprocessing: what partitions of the dataset are mean and variance computed over?
      train? train+validation?
    - Since the values are in kW, an hourly value would correspond to summing the values.
    - Testing: How exactly is the loss computed? From the description it can not be
      precisely inferred. Looking inside the code reveals that the test split is actually the
      last 14 days, which makes sense given that the observation period is 7 days. However,
      the paper does not mention the stride. Given the description
      "we use the past week (i.e. 168 hours) to forecast over the next 24 hours."
      Does not tell the stride, i.e. how much the sliding window is moved. We assume this to be
      24h.
    - Is the loss computed on the original scale, or on the pre-processed (i.e. z-score normalized)
      scale? The code reveals that apparantly the loss is computed on the original scale!

    Paper
    -----

    - | Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
      | https://www.sciencedirect.com/science/article/pii/S0169207021000637

    Evaluation Protocol
    -------------------
        In accordance with [9], we use the past week (i.e. 168 hours) to
        forecast over the next 24 hours.

        Electricity: Per [9], we use 500k samples taken between 2014-01-01 to 2014-09-01 – using
        the first 90% for training, and the last 10% as a validation set. Testing is done over the
        7 days immediately following the training set – as described in [9, 32]. Given the large
        differences in magnitude between trajectories, we also apply z-score normalization
        separately to each entity for real-valued inputs. In line with previous work, we consider
        the electricity usage, day-of-week, hour-of-day and and a time index – i.e. the number of
        time steps from the first observation – as real-valued inputs, and treat the entity
        identifier as a categorical variable.

    Test-Metric
    -----------

        Evaluation: $q$-Risk ($q=50$ and $q=90$)

        .. math::

            q-Risk = 2\frac{
                ∑_{y_t} ∑_{τ} QL(y(t), ŷ(t-τ), q)
            }{
                ∑_y ∑_{τ} |y(t)|
            }

    Results
    -------

    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    | Model | ARIMA | ConvTrans | DSSM  | DeepAR | ETS   | MQRNN | Seq2Seq | TFT   | TRMF  |
    +=======+=======+===========+=======+========+=======+=======+=========+=======+=======+
    | P50   | 0.154 | 0.059     | 0.083 | 0.075  | 0.102 | 0.077 | 0.067   | 0.055 | 0.084 |
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    | P90   | 0.102 | 0.034     | 0.056 | 0.400  | 0.077 | 0.036 | 0.036   | 0.027 | NaN   |
    +-------+-------+-----------+-------+--------+-------+-------+---------+-------+-------+
    """

    KeyType = Literal["train", "test", "valid", "joint", "whole"]
    r"""Type Hint for index."""
    preprocessor: BaseEncoder

    def __init__(self):
        super().__init__()
        self.observation_period = pd.Timedelta("7d")
        self.forecasting_period = pd.Timedelta("1d")
        self.observation_horizon = 24 * 7
        self.forecasting_horizon = 24

        self.preprocessor = Standardizer()
        self.preprocessor.fit(self.splits["train"])

    @cached_property
    def boundaries(self) -> dict[str, pd.Timestamp]:
        r"""Start and end dates of the training, validation and test sets."""
        return {
            "start": pd.Timestamp("2014-01-01"),
            "train": pd.Timestamp("2014-08-08"),
            "valid": pd.Timestamp("2014-09-01"),
            "final": pd.Timestamp("2014-09-08"),
        }

    @cached_property
    def test_metric(self) -> Callable[..., Tensor]:
        r"""Test metric."""

    @cached_property
    def dataset(self) -> pd.DataFrame:
        r"""Return the cached dataset."""
        ds = Electricity().dataset
        ds = ds.resample("1h").mean()
        mask = (self.boundaries["start"] <= ds.index) & (
            ds.index < self.boundaries["final"]
        )
        return ds[mask]

    @cached_property
    def index(self) -> Sequence[KeyType]:
        r"""List of entity identifiers."""
        return ["train", "test", "valid", "joint", "whole"]

    @cached_property
    def masks(self) -> dict[KeyType, np.ndarray]:
        r"""Masks for the training, validation and test sets."""
        return {
            "train": (self.boundaries["start"] <= self.dataset.index)
            & (self.dataset.index < self.boundaries["train"]),
            "valid": (
                self.boundaries["train"] - self.observation_period <= self.dataset.index
            )
            & (self.dataset.index < self.boundaries["valid"]),
            "test": (
                self.boundaries["valid"] - self.observation_period <= self.dataset.index
            )
            & (self.dataset.index < self.boundaries["final"]),
            "whole": (self.boundaries["start"] <= self.dataset.index)
            & (self.dataset.index < self.boundaries["final"]),
            "joint": (self.boundaries["start"] <= self.dataset.index)
            & (self.dataset.index < self.boundaries["valid"]),
        }

    @cached_property
    def splits(self) -> Mapping[KeyType, Any]:
        r"""Return cached splits of the dataset."""
        # We intentionally use these mask instead of the simpler
        # ds[lower:upper], in order to get the boundary inclusion right.

        return {
            "train": self.dataset[self.masks["train"]],
            "valid": self.dataset[self.masks["valid"]],
            "test": self.dataset[self.masks["test"]],
            "joint": self.dataset[self.masks["joint"]],
            "whole": self.dataset[self.masks["whole"]],
        }

    # @cached_property
    # def dataloader_kwargs(self) -> dict:
    #     r"""Return the kwargs for the dataloader."""
    #     return {
    #         "batch_size": 1,
    #         "shuffle": False,
    #         "sampler": None,
    #         "batch_sampler": None,
    #         "num_workers": 0,
    #         "collate_fn": lambda *x: x,
    #         "pin_memory": False,
    #         "drop_last": False,
    #         "timeout": 0,
    #         "worker_init_fn": None,
    #         "prefetch_factor": 2,
    #         "persistent_workers": False,
    #     }

    def get_dataloader(
        self, key: KeyType, /, shuffle: bool = False, **dataloader_kwargs: Any
    ) -> DataLoader:
        r"""Return the dataloader for the given key."""
        ds = self.splits[key]
        encoded = self.preprocessor.encode(ds)
        tensor = torch.tensor(encoded.values, dtype=torch.float32)

        sampler = SequenceSampler(
            encoded.index,
            stride="1d",
            seq_len=self.observation_period + self.forecasting_period,
            return_mask=True,
            shuffle=shuffle,
        )
        dataset = TensorDataset(tensor)

        return DataLoader(dataset, sampler=sampler, **dataloader_kwargs)


class ElectricityELBMBTTF:
    r"""Experiments as performed by the "LogSparseTransformer" paper.

    Paper
    -----
    - | Enhancing the Locality and Breaking the Memory Bottleneck of Transformer
        on Time Series Forecasting
      | Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, Xifeng Yan
      | Advances in Neural Information Processing Systems 32 (NeurIPS 2019)
      | http://proceedings.neurips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html

    Evaluation Protocol
    -------------------

        For short-term forecasting, we evaluate rolling-day forecasts for seven days ( i.e.,
        prediction horizon is one day and forecasts start time is shifted by one day after
        evaluating the prediction for the current day [6]). For long-term forecasting, we directly
        forecast 7 days ahead.

        A.2 Training
        [...]
        For electricity-c and traffic-c, we take 500K training windows while for electricity-f and
        traffic-f, we select 125K and 200K training windows, respectively.

        A.3 Evaluation
        Following the experimental settings in [6], one week data from 9/1/2014 00:00 (included) 9
        on electricity-c and 6/15/2008 17:00 (included) 10 on traffic-c is left as test sets.
        For electricity-f and traffic-f dataset, one week data from 8/31/2014 00:15 (included) and
        6/15/2008 17:00 (included) 11 is left as test sets, respectively.

    Test-Metric
    -----------

    R₀,₅ R₀,₉ losses

    Results
    -------

        Table 1: Results summary (R₀,₅/R₀,₉ -loss) of all methods. e-c and t-c represent
        electricity-c and traffic-c, respectively. In the 1st and 3rd row, we perform rolling-day
        prediction of 7 days while in the 2nd and 4th row, we directly forecast 7 days ahead.
        TRMF outputs points predictions, so we only report R₀,₅.


    +------+-------------+-------------+------------+-------------+-------------+-------------+
    |      | ARIMA       | ETS         | TRMF       | DeepAR      | DeepState   | Ours        |
    +======+=============+=============+============+=============+=============+=============+
    | e-c₁ | 0.154/0.102 | 0.101/0.077 | 0.084/---- | 0.075/0.040 | 0.083/0.056 | 0.059/0.034 |
    +------+-------------+-------------+------------+-------------+-------------+-------------+
    | e-c₇ | 0.283/0.109 | 0.121/0.101 | 0.087/---- | 0.082/0.053 | 0.085/0.052 | 0.070/0.044 |
    +------+-------------+-------------+------------+-------------+-------------+-------------+
    | t-c₁ | 0.223/0.137 | 0.236/0.148 | 0.186/---- | 0.161/0.099 | 0.167/0.113 | 0.122/0.081 |
    +------+-------------+-------------+------------+-------------+-------------+-------------+
    | t-c₇ | 0.492/0.280 | 0.509/0.529 | 0.202/---- | 0.179/0.105 | 0.168/0.114 | 0.139/0.094 |
    +------+-------------+-------------+------------+-------------+-------------+-------------+

    Fine (-f)

    +--------+----------------+-------------+
    |        | electricity-f₁ | traffic-f₁  |
    +========+================+=============+
    | DeepAR | 0.082/0.063    | 0.230/0.150 |
    +--------+----------------+-------------+
    | Ours   | 0.074/0.042    | 0.139/0.090 |
    +--------+----------------+-------------+
    """


class ElectricityNBEATS:
    r"""NBEATS."""


class ElectricityNHITS:
    r"""NHITS."""
