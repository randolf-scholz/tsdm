r"""Time Series and Time Series Collections.

This module provides wrappers for the datasets defined in `tsdm.datasets`.
"""

__all__ = [
    # Classes
    "DampedPendulum_Ansari2023",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "InSilico",
    "KiwiBenchmark",
    "MIMIC_III_DeBrouwer2019",
    "MIMIC_IV_Bilos2021",
    "PhysioNet2012",
    "PhysioNet2019",
    "Traffic",
    "USHCN",
    "USHCN_DeBrouwer2019",
]

from tsdm import datasets
from tsdm.timeseries.base import TimeSeries, TimeSeriesCollection


# region TimeSeries --------------------------------------------------------------------
def ETTh1() -> TimeSeries:
    r"""The ETTh1 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return TimeSeries(timeseries=ds["ETTh1"], name="ETTh1")


def ETTh2() -> TimeSeries:
    r"""The ETTh2 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return TimeSeries(timeseries=ds["ETTh2"], name="ETTh2")


def ETTm1() -> TimeSeries:
    r"""The ETTm1 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return TimeSeries(timeseries=ds["ETTm1"], name="ETTm1")


def ETTm2() -> TimeSeries:
    r"""The ETTm2 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return TimeSeries(timeseries=ds["ETTm2"], name="ETTm2")


def Electricity() -> TimeSeries:
    r"""The Electricity dataset wrapped as TimeSeriesCollection."""
    return TimeSeries.from_dataset(datasets.Electricity)


def Traffic() -> TimeSeries:
    r"""The Traffic dataset wrapped as TimeSeriesCollection."""
    return TimeSeries.from_dataset(datasets.Traffic)


# endregion TimeSeries -----------------------------------------------------------------


# region TimeSeriesCollection ----------------------------------------------------------
def InSilico() -> TimeSeriesCollection:
    r"""The in silico dataset wrapped as TimeSeriesCollection."""
    return TimeSeriesCollection.from_dataset(datasets.InSilico)


def KiwiBenchmark() -> TimeSeriesCollection:
    r"""The KIWI dataset wrapped as TimeSeriesCollection."""
    return TimeSeriesCollection.from_dataset(datasets.KiwiBenchmark)


def USHCN() -> TimeSeriesCollection:
    r"""The USHCN dataset wrapped as TimeSeriesCollection."""
    return TimeSeriesCollection.from_dataset(datasets.USHCN)


def USHCN_DeBrouwer2019() -> TimeSeriesCollection:
    r"""The USHCN_DeBrouwer2019 dataset wrapped as TimeSeriesCollection."""
    return TimeSeriesCollection.from_dataset(datasets.USHCN_DeBrouwer2019)


def PhysioNet2012() -> TimeSeriesCollection:
    r"""The PhysioNet2012 dataset wrapped as TimeSeriesCollection."""
    return TimeSeriesCollection.from_dataset(datasets.PhysioNet2012)


def PhysioNet2019() -> TimeSeriesCollection:
    r"""The PhysioNet2019 dataset wrapped as TimeSeriesCollection."""
    return TimeSeriesCollection.from_dataset(datasets.PhysioNet2019)


def MIMIC_IV_Bilos2021() -> TimeSeriesCollection:
    r"""The MIMIC_IV_Bilos2021 dataset wrapped as TimeSeriesCollection."""
    return TimeSeriesCollection.from_dataset(datasets.MIMIC_IV_Bilos2021)


def MIMIC_III_DeBrouwer2019() -> TimeSeriesCollection:
    r"""The MIMIC_III_DeBrouwer2019 dataset wrapped as TimeSeriesCollection."""
    return TimeSeriesCollection.from_dataset(datasets.MIMIC_III_DeBrouwer2019)


def DampedPendulum_Ansari2023() -> TimeSeriesCollection:
    r"""The DampedPendulum_Ansari2023 dataset wrapped as TimeSeriesCollection."""
    return TimeSeriesCollection.from_dataset(datasets.DampedPendulum_Ansari2023)


# endregion TimeSeriesCollection -------------------------------------------------------
