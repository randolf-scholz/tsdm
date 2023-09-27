"""Tasks based on the KIWI dataset."""

__all__ = [
    "InSilicoSampleGenerator",
    "InSilicoTask",
    "KiwiBenchmark",
    # OLD Tasks
    "Kiwi_BioProcessTask",
    "KIWI_FINAL_PRODUCT",
    "KIWI_RUNS_TASK",
    "KIWI_RUNS_GENERATOR",
]


from tsdm.tasks.kiwi._deprecated_kiwi_bioprocess import Kiwi_BioProcessTask
from tsdm.tasks.kiwi._deprecated_kiwi_runs_task import (
    KIWI_RUNS_GENERATOR,
    KIWI_RUNS_TASK,
)
from tsdm.tasks.kiwi.insilico import InSilicoSampleGenerator, InSilicoTask
from tsdm.tasks.kiwi.kiwi_benchmark import KiwiBenchmark
from tsdm.tasks.kiwi.kiwi_final_product import KIWI_FINAL_PRODUCT
