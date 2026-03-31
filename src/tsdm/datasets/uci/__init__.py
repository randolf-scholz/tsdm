r"""Datasets from the UCI machine learning repository.

References:
- `UCI Machine Learning Repository <https://archive.ics.uci.edu/>`_
"""

__all__ = [
    "Electricity",
    "Traffic",
    "BeijingAirQuality",
]

from .beijing_air_quality import BeijingAirQuality
from .electricity import Electricity
from .traffic import Traffic
