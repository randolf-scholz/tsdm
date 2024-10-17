r"""Time Series and Time Series Collections.

This module provides wrappers for the datasets defined in `tsdm.datasets`.
"""

__all__ = [
    # Classes
    "damped_pendulum_ansari2023",
    "etth1",
    "etth2",
    "ettm1",
    "ettm2",
    "electricity",
    "in_silico",
    "kiwi_benchmark",
    "mimic_iii_de_brouwer2019",
    "mimic_iv_bilos2021",
    "physio_net2012",
    "physio_net2019",
    "traffic",
    "ushcn",
    "ushcn_de_brouwer2019",
]

from tsdm import datasets
from tsdm.timeseries.base import PandasTS, PandasTSC


# region TimeSeries --------------------------------------------------------------------
def etth1() -> PandasTS:
    r"""The ETTh1 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return PandasTS(timeseries=ds["ETTh1"], name="ETTh1")


def etth2() -> PandasTS:
    r"""The ETTh2 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return PandasTS(timeseries=ds["ETTh2"], name="ETTh2")


def ettm1() -> PandasTS:
    r"""The ETTm1 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return PandasTS(timeseries=ds["ETTm1"], name="ETTm1")


def ettm2() -> PandasTS:
    r"""The ETTm2 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return PandasTS(timeseries=ds["ETTm2"], name="ETTm2")


def electricity() -> PandasTS:
    r"""The Electricity dataset wrapped as TimeSeriesCollection."""
    return PandasTS.from_dataset(datasets.Electricity)


def traffic() -> PandasTS:
    r"""The Traffic dataset wrapped as TimeSeriesCollection."""
    return PandasTS.from_dataset(datasets.Traffic)


# endregion TimeSeries -----------------------------------------------------------------


# region TimeSeriesCollection ----------------------------------------------------------
def in_silico() -> PandasTSC:
    r"""The in silico dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.InSilico)


def kiwi_benchmark() -> PandasTSC:
    r"""The KIWI dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.KiwiBenchmark)


def ushcn() -> PandasTSC:
    r"""The USHCN dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.USHCN)


def ushcn_de_brouwer2019() -> PandasTSC:
    r"""The USHCN_DeBrouwer2019 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.USHCN_DeBrouwer2019)


def physio_net2012() -> PandasTSC:
    r"""The PhysioNet2012 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.PhysioNet2012)


def physio_net2019() -> PandasTSC:
    r"""The PhysioNet2019 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.PhysioNet2019)


def mimic_iv_bilos2021() -> PandasTSC:
    r"""The MIMIC_IV_Bilos2021 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.MIMIC_IV_Bilos2021)


def mimic_iii_de_brouwer2019() -> PandasTSC:
    r"""The MIMIC_III_DeBrouwer2019 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.MIMIC_III_DeBrouwer2019)


def damped_pendulum_ansari2023() -> PandasTSC:
    r"""The DampedPendulum_Ansari2023 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.DampedPendulum_Ansari2023)


# endregion TimeSeriesCollection -------------------------------------------------------
