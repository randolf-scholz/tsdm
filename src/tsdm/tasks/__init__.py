r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a key to cater to the question of
forecasting horizons.


Idea:

The Pre-Encoder must work in the following way:

- `tuple[TimeTensor] → tuple[TimeTensor]` row-wise!
- `tuple[Tensor] → tuple[Tensor]`

More generally, eligible inputs are:

- `DataFrame`, `TimeTensor`, `tuple[DataFrame]`, `tuple[TimeTensor]`
- Product-types.

Must return a `NamedTuple` that agrees with the original column names!
This allows us to select
"""

__all__ = [
    # Sub- Modules
    "base",
    # Constants
    "Task",
    "TASKS",
    # Classes
    # Tasks
    "ETDatasetTask_Informer",
    "KIWI_RUNS_TASK",
    "KIWI_FINAL_PRODUCT",
    "ElectricityTFT",
    "MIMIC_III_DeBrouwer2019",
    "MIMIC_IV_Bilos2021",
    "USHCN_DeBrouwer2019",
    "Kiwi_BioProcessTask",
]


from typing import Final, TypeAlias

from tsdm.tasks import base
from tsdm.tasks.base import BaseTask
from tsdm.tasks.electricity import ElectricityTFT
from tsdm.tasks.etdataset import ETDatasetTask_Informer
from tsdm.tasks.kiwi_bioprocess import Kiwi_BioProcessTask
from tsdm.tasks.kiwi_final_product import KIWI_FINAL_PRODUCT
from tsdm.tasks.kiwi_runs_task import KIWI_RUNS_TASK
from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
from tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019

Task: TypeAlias = BaseTask
r"""Type hint for tasks."""

TASKS: Final[dict[str, type[Task]]] = {
    "ETDatasetTask_Informer": ETDatasetTask_Informer,
    "KIWI_RUNS_TASK": KIWI_RUNS_TASK,
    "KIWI_FINAL_PRODUCT": KIWI_FINAL_PRODUCT,
    "Kiwi_BioProcessTask": Kiwi_BioProcessTask,
    "ElectricityTFT": ElectricityTFT,
    "MIMIC_DeBrouwer": MIMIC_III_DeBrouwer2019,
    "MIMIC_IV_Bilos": MIMIC_IV_Bilos2021,
    "USHCN_DeBrouwer": USHCN_DeBrouwer2019,
}
r"""Dictionary of all available tasks."""
