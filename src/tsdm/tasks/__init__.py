r"""Tasks combine datasets with reproducible evaluation protocols.

A task provides dataset partitions, sample generators, samplers, data loaders,
and an error function. Samplers select sample identifiers; generators use those
identifiers to assemble inputs and targets. Collate functions combine samples
into batches and may apply encoders fitted on the associated training partition.

For forecasting, the observation horizon determines the available context and
the forecasting horizon determines the target interval.
"""

__all__ = [
    # Sub- Modules
    "base",
    "kiwi",
    "mimic",
    "synthetic",
    "uci",
    "ushcn",
    # Constants
    "TASKS",
    # ABCs & Protocols
    "Task",
    "TimeSeriesTask",
    # Classes
    "ETT_Zhou2021",
    "MIMIC_IV_Bilos2021",
    "USHCN_DeBrouwer2019",
    # New Tasks
    "DampedPendulum_Ansari2023",
    "InSilicoTask",
    "KiwiBenchmark",
]

from . import base, kiwi, mimic, synthetic, uci, ushcn
from .base import Task, TimeSeriesTask
from .ett.ett_zhou2021 import ETT_Zhou2021
from .kiwi import InSilicoTask, KiwiBenchmark
from .mimic import MIMIC_IV_Bilos2021
from .synthetic import DampedPendulum_Ansari2023
from .ushcn import USHCN_DeBrouwer2019

TASKS: dict[str, type[Task]] = {
    "KiwiBenchmark"             : KiwiBenchmark,
    "InSilicoTask"              : InSilicoTask,
    "DampedPendulum_Ansari2023" : DampedPendulum_Ansari2023,
}  # fmt: skip
r"""Dictionary of all available tasks."""
