r"""Datasets from the UCI machine learning repository.

References:
- `UCI Machine Learning Repository <https://archive.ics.uci.edu/>`_
"""

__all__ = [
    "Electricity",
    "Traffic",
    "BeijingAirQuality",
]

from tsdm.datasets.uci.beijing_air_quality import BeijingAirQuality
from tsdm.datasets.uci.electricity import Electricity
from tsdm.datasets.uci.traffic import Traffic
