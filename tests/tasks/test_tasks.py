#!/usr/bin/env python
r"""Test converters to masked format etc."""

import logging

from tsdm.datasets import InSilicoData, TimeSeriesCollection, TimeSeriesDataset
from tsdm.random.samplers import HierarchicalSampler, SlidingWindowSampler
from tsdm.tasks.base import Sample, TimeSeriesSampleGenerator

__logger__ = logging.getLogger(__name__)


def test_time_series_dataset_task():
    r"""Test TimeSeriesDatasetTask."""
    logging.basicConfig(level=logging.INFO)

    # make dataset
    dataset = InSilicoData()
    TSC = TimeSeriesCollection(
        timeseries=dataset.timeseries,
    )

    # make sampler, generate key
    subsamplers = {
        key: SlidingWindowSampler(
            ds.timeseries.index, horizons=["2h", "1h"], stride="1h"
        )
        for key, ds in TSC.items()
    }
    sampler = HierarchicalSampler(TSC, subsamplers, shuffle=False)
    key = next(iter(sampler))

    # construct the task dataset
    targets = ["Biomass", "Product"]
    observables = ["Biomass", "Substrate", "Acetate", "DOTm"]
    covariates = ["Volume", "Feed"]

    task = TimeSeriesSampleGenerator(
        TSC,
        targets=targets,
        observables=observables,
        covariates=covariates,
        sample_format=("sparse", "sparse"),
    )
    sample = task[key]
    assert isinstance(sample, Sample)

    # test with TimeSeriesDataset
    TSD = TSC[16130]
    assert isinstance(TSD, TimeSeriesDataset)
    task = TimeSeriesSampleGenerator(
        TSD,
        targets=targets,
        observables=observables,
        covariates=covariates,
        sample_format=("sparse", "sparse"),
    )
    sample = task[key[1]]
    assert isinstance(sample, Sample)


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    test_time_series_dataset_task()


if __name__ == "__main__":
    _main()
