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
        ds = KIWI_Dataset()
        super().__init__(
            timeseries=ds.timeseries,
            metadata=ds.metadata,
            timeseries_description=ds.timeseries_description,
            metadata_description=ds.metadata_description,
        )


class USHCN(TimeSeriesCollection):
    r"""The USHCN dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = USHCN_Dataset()
        super().__init__(
            timeseries=ds.timeseries,
            metadata=ds.metadata,
            timeseries_description=ds.timeseries_description,
            metadata_description=ds.metadata_description,
        )
