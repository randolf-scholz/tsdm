r"""Implementation of the kiwi task."""

__all__ = [
    # Classes
    "InSilicoSampleGenerator",
    "InSilicoTask",
]


from pandas import DataFrame
from torch.utils.data import Sampler as TorchSampler

from tsdm.datasets import InSilicoData, TimeSeriesCollection
from tsdm.random.samplers import HierarchicalSampler, SlidingWindowSampler
from tsdm.tasks.base import TimeSeriesSampleGenerator, TimeSeriesTask
from tsdm.types.variables import KeyVar as K
from tsdm.utils.data import folds_as_frame, folds_as_sparse_frame, folds_from_groups


class InSilicoSampleGenerator(TimeSeriesSampleGenerator):
    r"""Sample generator for the KIWI dataset."""

    def __init__(self, dataset):
        super().__init__(
            dataset,
            targets=["Biomass", "Product"],
            observables=["Biomass", "Substrate", "Acetate", "DOTm"],
            covariates=["Volume", "Feed"],
        )


class InSilicoTask(TimeSeriesTask):
    r"""Task for the KIWI dataset."""
    dataset: TimeSeriesCollection

    observation_horizon: str = "2h"
    r"""The number of datapoints observed during prediction."""
    forecasting_horizon: str = "1h'"
    r"""The number of datapoints the model should forecast."""

    def __init__(self) -> None:
        ds = InSilicoData()
        dataset = TimeSeriesCollection(ds.timeseries)
        super().__init__(dataset)

    @staticmethod
    def default_test_metric(*, targets, predictions):
        pass

    # def make_encoder(self, key: K, /) -> ModularEncoder:
    #     ...

    def make_sampler(self, key: K, /) -> TorchSampler:
        split: TimeSeriesCollection = self.splits[key]
        subsamplers = {
            key: SlidingWindowSampler(tsd.index, horizons=["2h", "1h"], stride="1h")
            for key, tsd in split.items()
        }
        return HierarchicalSampler(  # type: ignore[return-value]
            split, subsamplers, shuffle=self.split_type(key) == "training"
        )

    def make_folds(self, /) -> DataFrame:
        # TODO: refactor code, use **fold_kwargs, move code to base class?!?
        idx = self.dataset.index
        df = idx.to_frame(index=False).set_index(idx.names)  # i.e. add dummy column
        groups = df.groupby(idx.names, sort=False).ngroup()
        folds = folds_from_groups(
            groups, seed=2022, num_folds=8, train=6, valid=1, test=1
        )
        df = folds_as_frame(folds)
        return folds_as_sparse_frame(df)

    def make_generator(self, key: K, /) -> InSilicoSampleGenerator:
        split = self.splits[key]
        return InSilicoSampleGenerator(split)
