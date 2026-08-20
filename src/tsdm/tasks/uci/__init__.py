r"""Tasks based on UCI datasets."""

__all__ = [
    "Electricity_Sundar2018",
    "Electricity_Salinas2020",
    "Electricity_Yu2016",
    "Electricity_Li2019",
    "ElectricityLim2021",
    "Traffic_Lim2021",
    "Traffic_Yu2016",
]

from .electricity import (
    Electricity_Li2019,
    Electricity_Salinas2020,
    Electricity_Sundar2018,
    Electricity_Yu2016,
)
from .electricity_lim2021 import ElectricityLim2021
from .traffic import Traffic_Lim2021, Traffic_Yu2016
