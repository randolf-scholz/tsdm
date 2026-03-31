r"""Tasks based on UCI datasets."""

__all__ = [
    "ElectricityDeepAR",
    "ElectricityDeepState",
    "ElectricityELBMBTTF",
    "ElectricityLim2021",
    "ElectricityTRMF",
    "TrafficTFT",
    "TrafficTRMF",
]

from .electricity import (
    ElectricityDeepAR,
    ElectricityDeepState,
    ElectricityELBMBTTF,
    ElectricityTRMF,
)
from .electricity_lim2021 import ElectricityLim2021
from .traffic import TrafficTFT, TrafficTRMF
