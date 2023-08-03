"""Time series dataset."""

__all__ = [
    "KIWI",
    "PhysioNet2012",
    "USHCN",
    "USHCN_DeBrouwer2019",
]

from tsdm.datasets.base import TimeSeriesCollection
from tsdm.datasets.kiwi_benchmark import KIWI_Dataset
from tsdm.datasets.physionet2012 import PhysioNet2012 as _PhysioNet2012
from tsdm.datasets.ushcn import USHCN_Dataset
from tsdm.datasets.ushcn_debrouwer2019 import (
    USHCN_DeBrouwer2019 as _USHCN_DeBrouwer2019,
)


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


class USHCN_DeBrouwer2019(TimeSeriesCollection):
    r"""The USHCN dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = _USHCN_DeBrouwer2019()
        super().__init__(
            timeseries=ds.table,
        )


class PhysioNet2012(TimeSeriesCollection):
    r"""The PhysioNet2012 dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = _PhysioNet2012()
        super().__init__(
            timeseries=ds.timeseries,
            metadata=ds.metadata,
            timeseries_description=ds.timeseries_description,
            metadata_description=ds.metadata_description,
        )
