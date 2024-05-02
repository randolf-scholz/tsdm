r"""Datasets from the KIWI project.

References:
- `KIWI project <https://kiwi-biolab.de/>`_
"""

__all__ = [
    "InSilico",
    "InSilicoTSC",
    "KiwiBenchmark",
    "KiwiBenchmarkTSC",
    # OLD Datasets
    "KiwiRuns",
    "KiwiRunsTSC",
]

from tsdm.datasets.kiwi.in_silico import InSilico, InSilicoTSC
from tsdm.datasets.kiwi.kiwi_benchmark import KiwiBenchmark, KiwiBenchmarkTSC
from tsdm.datasets.kiwi.kiwi_runs import KiwiRuns, KiwiRunsTSC
