r"""Forecasting task on synthetic damped pendulum data.

References:
    - Neural Continuous-Discrete State Space Models
      Abdul Fatir Ansari, Alvin Heng, Andre Lim, Harold Soh
      Proceedings of the 40th International Conference on Machine Learning
      https://proceedings.mlr.press/v202/ansari23a.html
      https://github.com/clear-nus/NCDSSM
"""

__all__ = ["DampedPendulum_Ansari2023"]

from typing import Literal, cast, final

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from tsdm.datatools import CallableDataset, folds_as_frame, is_partition
from tsdm.prediction.pandas import SplitTimeData, make_sample_factory
from tsdm.random.samplers import (
    HierarchicalDataset,
    HierarchicalSampler,
    RandomSampler,
    Sampler,
)
from tsdm.tasks.base import TimeSeriesTask
from tsdm.timeseries.pandas import PandasTSC, damped_pendulum_ansari2023

type SplitID = tuple[int, Literal["train", "test", "valid"]]
type SampleKey = tuple[int, list[slice]]


@final
class DampedPendulum_Ansari2023(TimeSeriesTask[SplitID, SampleKey, SplitTimeData]):
    r"""Forecasting task on synthetic damped pendulum data.

    Note:
        - This uses scipy's builtin RK45 solver, instead of RK4.
        - The individual time series are only 150 steps long, so the whole sequence is returned

    .. epigraph::

        [section 5.1] We trained all the models on 10s/5s sequences (with a discretization of 0.1s)
        for bouncing ball/damped pendulum with 0%, 30%, 50% and 80% timesteps missing at random to
        simulate irregularly-sampled data. The models were evaluated on imputation of the missing
        timesteps and forecasts of 20s/10s beyond the training regime for bouncing ball/damped pendulum.

        [table 1] Imputation and forecasting results for bouncing ball and damped pendulum datasets
        averaged over 50 sample trajectories. Mean ± standard deviation are computed over 5 independent runs.

        [Appendix C.1] The training, validation, and test datasets consist of 5000, 1000, and 1000
        sequences of length 15s each, respectively.
    """

    num_folds: int = 5
    train_size: int = 5000
    valid_size: int = 1000
    test_size: int = 1000
    missing_rate: float = 0.0

    observation_horizon: float = 10.0
    prediction_horizon: float = 5.0

    def __init__(
        self,
        *,
        validate: bool = True,
        initialize: bool = True,
        missing_rate: float = 0.0,
    ) -> None:
        timeseries = damped_pendulum_ansari2023()
        super().__init__(dataset=timeseries, validate=validate, initialize=initialize)
        self.missing_rate = float(missing_rate)

    def make_generator(
        self, key: SplitID, /
    ) -> CallableDataset[SampleKey, SplitTimeData]:
        split = cast("PandasTSC", self.splits[key])
        columns = split.timeseries.columns
        return make_sample_factory(
            split,
            observables=columns,
            targets=columns,
            covariates=[],
        )

    def make_sampler(self, key: SplitID, /) -> Sampler[SampleKey]:
        split = self.splits[key]
        horizons = [
            slice(None, self.observation_horizon),
            slice(
                self.observation_horizon,
                self.observation_horizon + self.prediction_horizon,
            ),
        ]
        sampling_data = HierarchicalDataset(
            {series_key: [horizons] for series_key in split}
        )
        subsamplers = {series_key: RandomSampler([horizons]) for series_key in split}
        return HierarchicalSampler(
            sampling_data,
            subsamplers,
            shuffle=self.is_train_split(key),
        )

    def make_folds(self, /) -> DataFrame:
        r"""Create the folds."""
        # NOTE: all folds are the same due to fixed random state.
        # SEE: https://github.com/mbilos/neural-flows-experiments/blob/bd19f7c92461e83521e268c1a235ef845a3dd963/nfe/experiments/gru_ode_bayes/lib/get_data.py#L66-L67
        folds = []
        for _ in range(self.num_folds):
            # get the test split
            train_idx, test_idx = train_test_split(
                self.dataset.metaindex,
                test_size=self.test_size,
            )
            # get the train and validation split
            train_idx, valid_idx = train_test_split(
                train_idx,
                test_size=self.valid_size,
            )
            fold = {
                "train": train_idx,
                "valid": valid_idx,
                "test": test_idx,
            }
            if not is_partition(fold.values(), union=self.dataset.metaindex):
                raise ValueError("Invalid partitions!")
            folds.append(fold)
        return folds_as_frame(folds, index=self.dataset.metaindex, sparse=True)
