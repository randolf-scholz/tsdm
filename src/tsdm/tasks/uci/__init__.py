"""Tasks based on UCI datasets."""

__all__ = [
    "ElectricityDeepAR",
    "ElectricityDeepState",
    "ElectricityELBMBTTF",
    "ElectricityLim2021",
    "ElectricityTRMF",
    "TrafficTFT",
    "TrafficTRMF",
]

from tsdm.tasks.uci.electricity import (
    ElectricityDeepAR,
    ElectricityDeepState,
    ElectricityELBMBTTF,
    ElectricityTRMF,
)
from tsdm.tasks.uci.electricity_lim2021 import ElectricityLim2021
from tsdm.tasks.uci.traffic import TrafficTFT, TrafficTRMF
