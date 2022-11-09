r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "KiwiSampleGenerator",
    "KiwiTask",
]


from typing import Mapping

from torch.utils.data import Sampler as TorchSampler

from tsdm.datasets import Dataset, KiwiDataset, TimeSeriesCollection
from tsdm.random.samplers import HierarchicalSampler, SlidingWindowSampler
from tsdm.tasks.base import TimeSeriesSampleGenerator, TimeSeriesTask
from tsdm.utils.data import folds_as_frame, folds_as_sparse_frame, folds_from_groups
from tsdm.utils.types import KeyVar


class KiwiSampleGenerator(TimeSeriesSampleGenerator):
    r"""Sample generator for the KIWI dataset."""

    def __init__(self, dataset):
        super().__init__(
            dataset,
            observables=[
                "Base",
                "DOT",
                "Glucose",
                "OD600",
                "Acetate",
                "Fluo_GFP",
                "Volume",
                "pH",
            ],
            covariates=[
                "Cumulated_feed_volume_glucose",
                "Cumulated_feed_volume_medium",
                "InducerConcentration",
                "StirringSpeed",
                "Flow_Air",
                "Temperature",
                "Probe_Volume",
            ],
            targets=["Fluo_GFP"],
        )


class KiwiTask(TimeSeriesTask):
    r"""Task for the KIWI dataset."""
    # dataset: TimeSeriesCollection = KiwiDataset()
    observation_horizon: str = "2h"
    r"""The number of datapoints observed during prediction."""
    forecasting_horizon: str = "1h'"
    r"""The number of datapoints the model should forecast."""

    def __init__(self) -> None:
        super().__init__(dataset=KiwiDataset())

    @staticmethod
    def default_metric(*, targets, predictions):
        pass

    # def make_encoder(self, key: KeyVar, /) -> ModularEncoder:
    #     ...

    def make_sampler(self, key: KeyVar, /) -> TorchSampler:
        split: TimeSeriesCollection = self.splits[key]
        subsamplers = {
            key: SlidingWindowSampler(tsd.index, horizons=["2h", "1h"], stride="1h")
            for key, tsd in split.items()
        }
        return HierarchicalSampler(split, subsamplers, shuffle=False)  # type: ignore[return-value]

    def make_folds(self, /) -> Mapping[KeyVar, Dataset]:
        md = self.dataset.metadata
        groups = md.groupby(["run_id", "color"], sort=False).ngroup()
        folds = folds_from_groups(
            groups, seed=2022, num_folds=5, train=7, valid=1, test=2
        )
        df = folds_as_frame(folds)
        return folds_as_sparse_frame(df)

    def make_generator(self, key: KeyVar, /) -> KiwiSampleGenerator:
        split = self.splits[key]
        return KiwiSampleGenerator(split)
