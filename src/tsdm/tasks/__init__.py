r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object. We decided to use a dataloader instead of, say, a key to cater to the question of
forecasting horizons.

.. note::

    One thing that is weird about torch's way to approach the issue, is that there
    are two seperated functionalities: sampling rows and selecting columns.
    In principle, the sampler could do both, supposedly.
    However, the TaskDataset is really responsible for creating the sample.

.. admonition:: Idea

    The Pre-Encoder must work in the following way:

    - `tuple[TimeTensor] → tuple[TimeTensor]` row-wise!
    - `tuple[Tensor] → tuple[Tensor]`.


More generally, eligible inputs are:

- `DataFrame`, `TimeTensor`, `tuple[DataFrame]`, `tuple[TimeTensor]`
- Product-types.

Must return a `NamedTuple` that agrees with the original column names!
"""

__all__ = [
    # Sub- Modules
    "base",
    "kiwi",
    "mimic",
    "uci",
    # Protocol
    "Task",
    # Constants
    "TASKS",
    # Classes
    "OldBaseTask",
    "TimeSeriesTask",
    "TimeSeriesSampleGenerator",
    # Tasks
    "ETT_Zhou2021",
    "ElectricityDeepAR",
    "ElectricityDeepState",
    "ElectricityELBMBTTF",
    "ElectricityLim2021",
    "ElectricityTRMF",
    "KIWI_FINAL_PRODUCT",
    "KiwiForecastingTask",
    "Kiwi_BioProcessTask",
    "MIMIC_III_Bilos2021",
    "MIMIC_III_DeBrouwer2019",
    "MIMIC_IV_Bilos2021",
    "TrafficTFT",
    "TrafficTRMF",
    "USHCN_DeBrouwer2019",
    # Task Datasets
    "InSilicoSampleGenerator",
    "InSilicoTask",
    "KiwiTask",
]

from tsdm.tasks import base, kiwi, mimic, uci
from tsdm.tasks._deprecated import OldBaseTask
from tsdm.tasks.base import Task, TimeSeriesSampleGenerator, TimeSeriesTask
from tsdm.tasks.ett_zhou2021 import ETT_Zhou2021
from tsdm.tasks.kiwi import (
    KIWI_FINAL_PRODUCT,
    InSilicoSampleGenerator,
    InSilicoTask,
    Kiwi_BioProcessTask,
    KiwiForecastingTask,
    KiwiTask,
)
from tsdm.tasks.mimic import (
    MIMIC_III_Bilos2021,
    MIMIC_III_DeBrouwer2019,
    MIMIC_IV_Bilos2021,
)
from tsdm.tasks.uci import (
    ElectricityDeepAR,
    ElectricityDeepState,
    ElectricityELBMBTTF,
    ElectricityLim2021,
    ElectricityTRMF,
    TrafficTFT,
    TrafficTRMF,
)
from tsdm.tasks.ushcn import USHCN_DeBrouwer2019

TASKS: dict[str, type[Task]] = {
    "ETT_Zhou2021": ETT_Zhou2021,
    "ElectricityLim2021": ElectricityLim2021,
    "KIWI_FINAL_PRODUCT": KIWI_FINAL_PRODUCT,
    "KiwiTask": KiwiTask,
    "Kiwi_BioProcessTask": Kiwi_BioProcessTask,
    "MIMIC_III_Bilos2021": MIMIC_III_DeBrouwer2019,
    "MIMIC_III_DeBrouwer2019": MIMIC_III_DeBrouwer2019,
    "MIMIC_IV_Bilos2021": MIMIC_IV_Bilos2021,
    "USHCN_DeBrouwer2019": USHCN_DeBrouwer2019,
}
r"""Dictionary of all available tasks."""
