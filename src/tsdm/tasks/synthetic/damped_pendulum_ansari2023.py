"""Forecasting task on synthetic damped pendulum data.

References:
    - Neural Continuous-Discrete State Space Models
      Abdul Fatir Ansari, Alvin Heng, Andre Lim, Harold Soh
      Proceedings of the 40th International Conference on Machine Learning
      https://proceedings.mlr.press/v202/ansari23a.html
      https://github.com/clear-nus/NCDSSM
"""

__all__ = ["DampedPendulum_Ansari2023"]

from typing import final

from pandas import DataFrame
from torch.utils.data import Sampler as TorchSampler

from tsdm import datasets
from tsdm.tasks.base import SplitID, TimeSeriesSampleGenerator, TimeSeriesTask
from tsdm.types.variables import key_var as K


@final
class DampedPendulum_Ansari2023(TimeSeriesTask):
    """Forecasting task on synthetic damped pendulum data."""

    def __init__(self) -> None:
        dataset = datasets.synthetic.DampedPendulum_Ansari2023()
        timeseries = datasets.TimeSeriesCollection(timeseries=dataset.table)

        super().__init__(dataset=timeseries)

    def make_generator(self, key: SplitID, /) -> TimeSeriesSampleGenerator:
        raise NotImplementedError

    def make_sampler(self, key: SplitID, /) -> TorchSampler[K]:
        raise NotImplementedError

    def make_folds(self, /) -> DataFrame:
        raise NotImplementedError
