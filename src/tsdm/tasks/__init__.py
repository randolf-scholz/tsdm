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
    # Constants
    "Task",
    "TASKS",
    # Classes
    "TimeSeriesTask",
    "TimeSeriesSampleGenerator",
    # Tasks
    "ETT_Zhou2021",
    "KIWI_RUNS_TASK",
    "KIWI_FINAL_PRODUCT",
    "ElectricityLim2021",
    "MIMIC_III_DeBrouwer2019",
    "MIMIC_III_Bilos2021",
    "MIMIC_IV_Bilos2021",
    "USHCN_DeBrouwer2019",
    "Kiwi_BioProcessTask",
    # Task Datasets
    "KiwiTask",
    "InSilicoSampleGenerator",
    "InSilicoTask",
]


from typing import Final, TypeAlias

from tsdm.tasks import base
from tsdm.tasks._deprecated import OldBaseTask
from tsdm.tasks._deprecated_electricity_lim2021 import ElectricityLim2021
from tsdm.tasks._deprecated_kiwi_bioprocess import Kiwi_BioProcessTask
from tsdm.tasks._deprecated_kiwi_runs_task import KIWI_RUNS_TASK
from tsdm.tasks.base import TimeSeriesSampleGenerator, TimeSeriesTask
from tsdm.tasks.ett_zhou2021 import ETT_Zhou2021
from tsdm.tasks.insilico import InSilicoSampleGenerator, InSilicoTask
from tsdm.tasks.kiwi_final_product import KIWI_FINAL_PRODUCT
from tsdm.tasks.kiwi_task import KiwiTask
from tsdm.tasks.mimic_iii_bilos2021 import MIMIC_III_Bilos2021
from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
from tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019

Task: TypeAlias = OldBaseTask
r"""Type hint for tasks."""

TASKS: Final[dict[str, type[Task]]] = {
    "ETT_Zhou2021": ETT_Zhou2021,
    "KIWI_RUNS_TASK": KIWI_RUNS_TASK,
    "KIWI_FINAL_PRODUCT": KIWI_FINAL_PRODUCT,
    "Kiwi_BioProcessTask": Kiwi_BioProcessTask,
    "ElectricityLim2021": ElectricityLim2021,
    "MIMIC_III_DeBrouwer2019": MIMIC_III_DeBrouwer2019,
    "MIMIC_III_Bilos2021": MIMIC_III_DeBrouwer2019,
    "MIMIC_IV_Bilos2021": MIMIC_IV_Bilos2021,
    "USHCN_DeBrouwer2019": USHCN_DeBrouwer2019,
}
r"""Dictionary of all available tasks."""

del Final, TypeAlias
