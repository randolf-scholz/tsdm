"""Time series dataset."""

__all__ = [
    "KIWI",
    "USHCN",
]

from tsdm.datasets.base import TimeSeriesCollection
from tsdm.datasets.kiwi_benchmark import KIWI_Dataset
from tsdm.datasets.ushcn import USHCN_Dataset


class KIWI(TimeSeriesCollection):
    r"""The KIWI dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        super().__init__(**KIWI_Dataset())


class USHCN(TimeSeriesCollection):
    r"""The USHCN dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        super().__init__(**USHCN_Dataset())
