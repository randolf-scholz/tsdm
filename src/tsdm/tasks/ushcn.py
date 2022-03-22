r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = ["USHCN_DeBrouwer"]

import logging
from collections.abc import Callable
from functools import cached_property
from typing import Any, Literal

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader

from tsdm.datasets import USHCN_SmallChunkedSporadic
from tsdm.encoders import FunctionalEncoders
from tsdm.losses import ModularLoss, ModularLosses
from tsdm.tasks.base import BaseTask
from tsdm.util import Split

__logger__ = logging.getLogger(__name__)


class USHCN_DeBrouwer(BaseTask):
    r"""Preprocessed subset of the USHCN climate dataset used by De Brouwer et. al.

    Evaluation Protocol
    -------------------

        5.3Climate forecast

        From short-term weather forecast to long-range prediction or assessment of systemic
        changes, such as global warming, climatic data has always been a popular application for
        time-series analysis. This data is often considered to be regularly sampled over long
        periods of time, which facilitates their statistical analysis. Yet, this assumption does
        not usually hold in practice. Missing data are a problem that is repeatedly encountered in
        climate research because of, among others, measurement errors, sensor failure, or faulty
        data acquisition. The actual data is then sporadic and researchers usually resort to
        imputation before statistical analysis (Junninen et al., 2004; Schneider, 2001).

        We use the publicly available United State Historical Climatology Network (USHCN) daily
        data set (Menne et al.), which contains measurements of 5 climate variables
        (daily temperatures, precipitation, and snow) over 150 years for 1,218 meteorological
        stations scattered over the United States. We selected a subset of 1,114 stations and an
        observation window of 4 years (between 1996 and 2000). To make the time series sporadic, we
        subsample the data such that each station has an average of around 60 observations over
        those 4 years. Appendix L contains additional details regarding this procedure.
        The task is then to predict the next 3 measurements after the first 3 years of observation.

    References
    ----------
    - | `GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series
        <https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html>`_
      | De Brouwer, Edward and Simm, Jaak and Arany, Adam and Moreau, Yves
      | `Advances in Neural Information Processing Systems 2019
        <https://proceedings.neurips.cc/paper/2019>`_
    """

    @cached_property
    def index(self) -> list[str]:
        """TODO: Add index."""

    test_metric = type[ModularLoss]

    splits: dict[int, dict[str, DataFrame]]
    folds: list[Split]
    observation_horizon: int
    forecasting_horizon: int
    accumulation_function: Callable[..., Tensor]

    @cached_property
    def dataset(self) -> USHCN_SmallChunkedSporadic:
        r"""Return cached dataset."""
        return USHCN_SmallChunkedSporadic()

    def __init__(
        self,
        test_metric: Literal["MSE", "MAE"] = "MSE",
        time_encoder: str = "time2float",
    ) -> None:
        super().__init__()
        self._gen_folds()
        self.forecasting_horizon = 3
        self.observation_horizon = 3
        self.test_metric = ModularLosses[test_metric]
        self.time_encoder = FunctionalEncoders[time_encoder]
        self.horizon = self.observation_horizon + self.forecasting_horizon
        self.accumulation_function = nn.Identity()  # type: ignore[assignment]

    def _gen_folds(self) -> None:
        N = self.dataset.dataset["ID"].nunique()
        num_folds = 5
        np.random.seed(432)

        folds = []
        for _ in range(num_folds):
            train_idx, test_idx = train_test_split(np.arange(N), test_size=0.1)
            train_idx, val_idx = train_test_split(train_idx, test_size=0.2)
            folds.append(Split(train=train_idx, valid=val_idx, test=test_idx))

        self.folds = folds

    def get_dataloader(
        self,
        key: str,
        /,
        shuffle: bool = False,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split & fold.

        Parameters
        ----------
        key: str
            From which part of the dataset to construct the loader
        shuffle: bool = True

        Returns
        -------
        DataLoader
        """
        if key == "test":
            assert not shuffle, "Don't shuffle when evaluating test-dataset!"
        if key == "test" and "drop_last" in dataloader_kwargs:
            assert not dataloader_kwargs[
                "drop_last"
            ], "Don't drop when evaluating test-dataset!"

        # the split = self.folds[fold]
        raise NotImplementedError
