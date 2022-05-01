r"""Dataset Import Facilities.

Implement your own by subclassing `BaseDataset`

Basic Usage
-----------

.. code-block:: python

   from tsdm.dataset import Electricity

   print(vars(Electricity))
   Electricity.download()
   Electricity.preprocess()
   x = Electricity.load()

   # or, simply:
   x = Electricity.dataset


Some design decisions:

1. Why not use `Series` instead of Mapping for dataset?
    - `Series[object]` has bad performance issues in construction.

2. Should we have Dataset style iteration or dict style iteration?
    - Note that for `dict`, `iter(dict)` iterates over index.
    - For `Series`, `DataFrame`, `TorchDataset`, `__iter__` iterates over values.
"""

__all__ = [
    # Sub-Packages
    "base",
    "synthetic",
    "torch",
    # Types
    "dataset",
    # Constants
    "DATASETS",
    # ABCs
    "BaseDataset",
    "Dataset",
    "Template",
    # Classes
    "IndexedArray",
    "TimeSeriesDataset",
    "TimeTensor",
    # Datasets
    "BeijingAirQuality",
    "ETT",
    "Electricity",
    "InSilicoData",
    "KIWI_RUNS",
    "MIMIC_III",
    "Physionet2019",
    "Traffic",
    "USHCN",
    "USHCN_SmallChunkedSporadic",
]


import logging
from typing import Final

from tsdm.datasets import base, synthetic, torch
from tsdm.datasets.base import BaseDataset, Dataset, Template
from tsdm.datasets.beijing_air_quality import BeijingAirQuality
from tsdm.datasets.electricity import Electricity
from tsdm.datasets.ett import ETT
from tsdm.datasets.in_silico_data import InSilicoData
from tsdm.datasets.kiwi_runs import KIWI_RUNS
from tsdm.datasets.mimic_iii import MIMIC_III
from tsdm.datasets.physionet2019 import Physionet2019
from tsdm.datasets.torch import IndexedArray, TimeSeriesDataset, TimeTensor
from tsdm.datasets.traffic import Traffic
from tsdm.datasets.ushcn import USHCN, USHCN_SmallChunkedSporadic

__logger__ = logging.getLogger(__name__)

dataset = BaseDataset
r"""Type hint for dataset."""

DATASETS: Final[dict[str, type[dataset]]] = {
    "BeijingAirQuality": BeijingAirQuality,
    "ETT": ETT,
    "Electricity": Electricity,
    "InSilicoData": InSilicoData,
    "KIWI_RUNS_TASK": KIWI_RUNS,
    "MIMIC_III": MIMIC_III,
    "Physionet2019": Physionet2019,
    "Traffic": Traffic,
    "USHCN": USHCN,
    "USHCN_SmallChunkedSporadic": USHCN_SmallChunkedSporadic,
}
r"""Dictionary of all available dataset."""
