r"""Test converters to masked format etc."""

import logging
from collections.abc import Sequence

import numpy as np
from pytest import mark

from tsdm.data import TimeSeriesSampleGenerator
from tsdm.data.timeseries import Sample, TimeSeriesCollection, TimeSeriesDataset
from tsdm.datasets import InSilico
from tsdm.random.samplers import HierarchicalSampler, SlidingSampler
from tsdm.tasks import (
    MIMIC_III_Bilos2021,
    MIMIC_III_DeBrouwer2019,
    MIMIC_IV_Bilos2021,
    USHCN_DeBrouwer2019,
)

__logger__ = logging.getLogger(__name__)

TASKS = {
    "MIMIC_III_Bilos2021": MIMIC_III_Bilos2021,
    "MIMIC_III_DeBrouwer2019": MIMIC_III_DeBrouwer2019,
    "MIMIC_IV_Bilos2021": MIMIC_IV_Bilos2021,
    "USHCN_DeBrouwer2019": USHCN_DeBrouwer2019,
}


@mark.slow
@mark.parametrize("task_cls", TASKS.values(), ids=TASKS.keys())
def test_tasks(task_cls: type) -> None:
    task = task_cls()
    key = (0, "train")
    dataloader = task.make_dataloader(key)
    next(iter(dataloader))


def test_time_series_sample_generator() -> None:
    """Test the TimeSeriesSampleGenerator."""
    LOGGER = __logger__.getChild(TimeSeriesSampleGenerator.__name__)
    LOGGER.info("Testing.")

    # make dataset
    dataset = InSilico()
    TSC = TimeSeriesCollection(
        timeseries=dataset.timeseries,
    )

    # make sampler, generate key
    subsamplers = {
        key: SlidingSampler(
            ds.timeseries.index,
            horizons=["2h", "1h"],
            stride="1h",
            mode="masks",
        )
        for key, ds in TSC.items()
    }
    sampler = HierarchicalSampler(TSC, subsamplers, shuffle=False)
    key = next(iter(sampler))

    # validate key
    assert isinstance(key, tuple) and len(key) == 2
    assert key[0] == 16130
    assert isinstance(key[1], Sequence)
    assert all(isinstance(horizon, np.ndarray) for horizon in key[1])

    # construct the sample generator
    targets = ["Biomass", "Product"]
    observables = ["Biomass", "Substrate", "Acetate", "DOTm"]
    covariates = ["Volume", "Feed"]

    generator = TimeSeriesSampleGenerator(
        TSC,
        targets=targets,
        observables=observables,
        covariates=covariates,
        sparse_index=True,
        sparse_columns=True,
    )
    sample = generator[key]
    assert isinstance(sample, Sample)

    # test with TimeSeriesDataset
    outer_key = key[0]
    inner_key = key[1]
    TSD = TSC[outer_key]  # selecting individual time series.
    assert isinstance(TSD, TimeSeriesDataset)

    generator = TimeSeriesSampleGenerator(
        TSD,
        targets=targets,
        observables=observables,
        covariates=covariates,
        sparse_index=True,
        sparse_columns=True,
    )

    # assert isinstance(subkey, tuple)

    sample = generator[inner_key]
    assert isinstance(sample, Sample)
