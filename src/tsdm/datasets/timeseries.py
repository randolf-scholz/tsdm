"""Time series dataset."""

__all__ = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "KIWI",
    "MIMIC_III_DeBrouwer2019",
    "MIMIC_IV_Bilos2021",
    "PhysioNet2012",
    "PhysioNet2019",
    "USHCN",
    "USHCN_DeBrouwer2019",
]

from tsdm.datasets.base import TimeSeriesCollection
from tsdm.datasets.ett import ETT
from tsdm.datasets.kiwi.kiwi_benchmark import KIWI_Dataset
from tsdm.datasets.mimic.mimic_iii_debrouwer2019 import (
    MIMIC_III_DeBrouwer2019 as _MIMIC_III_DeBrouwer2019,
)
from tsdm.datasets.mimic.mimic_iv_bilos2021 import (
    MIMIC_IV_Bilos2021 as _MIMIC_IV_Bilos2021,
)
from tsdm.datasets.physionet.physionet2012 import PhysioNet2012 as _PhysioNet2012
from tsdm.datasets.physionet.physionet2019 import PhysioNet2019 as _PhysioNet2019
from tsdm.datasets.uci.electricity import Electricity as _Electricity
from tsdm.datasets.uci.traffic import Traffic as _Traffic
from tsdm.datasets.ushcn.ushcn import USHCN as _USHCN
from tsdm.datasets.ushcn.ushcn_debrouwer2019 import (
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
        ds = _USHCN()
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


class PhysioNet2019(TimeSeriesCollection):
    r"""The PhysioNet2019 dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = _PhysioNet2019()
        super().__init__(
            timeseries=ds.timeseries,
            metadata=ds.metadata,
            timeseries_description=ds.timeseries_description,
            metadata_description=ds.metadata_description,
        )


class MIMIC_IV_Bilos2021(TimeSeriesCollection):
    r"""The MIMIC_IV_Bilos2021 dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = _MIMIC_IV_Bilos2021()
        super().__init__(timeseries=ds.table)


class MIMIC_III_DeBrouwer2019(TimeSeriesCollection):
    r"""The MIMIC_III_DeBrouwer2019 dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = _MIMIC_III_DeBrouwer2019()
        super().__init__(timeseries=ds.table)


class ETTh1(TimeSeriesCollection):
    r"""The ETTh1 dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = ETT()
        super().__init__(timeseries=ds["ETTh1"])


class ETTh2(TimeSeriesCollection):
    r"""The ETTh1 dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = ETT()
        super().__init__(timeseries=ds["ETTh2"])


class ETTm1(TimeSeriesCollection):
    r"""The ETTh1 dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = ETT()
        super().__init__(timeseries=ds["ETTm1"])


class ETTm2(TimeSeriesCollection):
    r"""The ETTh1 dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = ETT()
        super().__init__(timeseries=ds["ETTm2"])


class Electricity(TimeSeriesCollection):
    r"""The Electricity dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = _Electricity()
        super().__init__(timeseries=ds.table)


class Traffic(TimeSeriesCollection):
    r"""The Traffic dataset wrapped as TimeSeriesCollection."""

    def __init__(self) -> None:
        ds = _Traffic()
        super().__init__(timeseries=ds.table)
