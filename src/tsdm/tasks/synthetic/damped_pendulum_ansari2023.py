r"""Forecasting task on synthetic damped pendulum data.

References:
    - Neural Continuous-Discrete State Space Models <br>
      Abdul Fatir Ansari, Alvin Heng, Andre Lim, Harold Soh <br>
      Proceedings of the 40th International Conference on Machine Learning <br>
      https://proceedings.mlr.press/v202/ansari23a.html <br>
      https://github.com/clear-nus/NCDSSM
"""

__all__ = ["DampedPendulum_Ansari2023"]

from functools import cached_property
from typing import Literal, final

from pandas import DataFrame, Index, MultiIndex

from tsdm.datatools import CallableDataset
from tsdm.losses.sequence import mse_loss
from tsdm.random.samplers import HierarchicalSampler, RandomSampler, Sampler
from tsdm.tasks.base import TimeSeriesTask
from tsdm.timeseries.pandas import (
    SplitTimeData,
    TimeSeriesCollection,
    damped_pendulum_ansari2023,
    make_sample_factory,
)

type SplitID = tuple[int, Literal["train", "test", "valid"]]
type SampleKey = tuple[int, list[slice]]


@final
class DampedPendulum_Ansari2023(
    TimeSeriesTask[SplitID, TimeSeriesCollection[int], SampleKey, SplitTimeData]
):
    r"""Forecasting task on synthetic damped pendulum data.

    Note:
        - The data generator uses SciPy's built-in RK45 solver rather than RK4.
        - Each sampled trajectory is split into an observation window of 10 seconds
            and a prediction window of 5 seconds by default.
        - Fold keys represent repeated runs on one fixed train/validation/test split,
            rather than cross-validation folds.

    ### Evaluation Protocol

    > [section 5.1] We trained all the models on 10s/5s sequences (with a discretization of 0.1s)
    > for bouncing ball/damped pendulum with 0%, 30%, 50% and 80% timesteps missing at random to
    > simulate irregularly-sampled data. The models were evaluated on imputation of the missing
    > timesteps and forecasts of 20s/10s beyond the training regime for bouncing ball/damped pendulum.
    >
    > [table 1] Imputation and forecasting results for bouncing ball and damped pendulum datasets
    > averaged over 50 sample trajectories. Mean ± standard deviation are computed over 5 independent runs.
    >
    > [Appendix C.1] The training, validation, and test datasets consist of 5000, 1000, and 1000
    > sequences of length 15s each, respectively.

    References:
        - Neural Continuous-Discrete State Space Models <br>
          Abdul Fatir Ansari, Alvin Heng, Andre Lim, Harold Soh <br>
          Proceedings of the 40th International Conference on Machine Learning <br>
          https://proceedings.mlr.press/v202/ansari23a.html <br>
          https://github.com/clear-nus/NCDSSM
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
        missing_rate: float = 0.0,
    ) -> None:
        timeseries = damped_pendulum_ansari2023()
        self.missing_rate = float(missing_rate)
        super().__init__(
            dataset=timeseries,
            error_fn=lambda predictions, targets: mse_loss(
                predictions=predictions, targets=targets, time_dim=-2, channel_dim=-1
            ),
        )

    @property
    def split_keys(self) -> tuple[SplitID, ...]:
        return tuple(self.split_masks.columns)

    def is_train_split(self, key: SplitID) -> bool:
        return key[-1] == "train"

    def make_partition(self, key: SplitID, /) -> TimeSeriesCollection[int]:
        return self.dataset[self.split_masks[key].to_list()]

    def make_generator(
        self, key: SplitID, /
    ) -> CallableDataset[SampleKey, SplitTimeData]:
        split = self.partitions[key]
        columns = split.timeseries.columns
        return make_sample_factory(
            split,
            observables=columns,
            targets=columns,
            covariates=[],
        )

    def make_sampler(self, key: SplitID, /) -> Sampler[SampleKey]:
        split = self.partitions[key]
        horizons = [
            slice(None, self.observation_horizon),
            slice(
                self.observation_horizon,
                self.observation_horizon + self.prediction_horizon,
            ),
        ]
        sampling_data = {series_key: [horizons] for series_key in split}
        subsamplers = {series_key: RandomSampler([horizons]) for series_key in split}
        return HierarchicalSampler(
            sampling_data,
            subsamplers,
            shuffle=self.is_train_split(key),
        )

    @cached_property
    def split_masks(self) -> DataFrame:
        r"""Reuse consecutive train/validation/test blocks for every run.

        NCDSSM generates the three datasets consecutively under one seed (42 by
        default), saves them separately, and loads them without repartitioning.
        Our generated collection stores all trajectories together, so split it
        in the same order. This reproduces the split structure, not the exact
        trajectories produced by the reference generator.
        """
        # SEE: https://github.com/clear-nus/NCDSSM/blob/7212bba9616d81c948f507eef9eb1f7d5cf1954e/data/damped_pendulum.py#L267-L301
        index = Index(self.dataset.metaindex.to_list())
        train_stop = self.train_size
        valid_stop = train_stop + self.valid_size
        if len(index) != valid_stop + self.test_size:
            raise ValueError("Partition sizes must sum to the number of trajectories.")
        fold = {
            "train": index[:train_stop],
            "valid": index[train_stop:valid_stop],
            "test": index[valid_stop:],
        }
        fold_masks = {
            split: index.isin(members)
            for split, members in sorted(fold.items())
            if len(members)
        }
        masks = {
            (fold_id, split): mask
            for fold_id in range(self.num_folds)
            for split, mask in fold_masks.items()
        }
        columns = MultiIndex.from_tuples(list(masks), names=["fold", "partition"])
        return DataFrame(masks, index=index, columns=columns)
