r"""MIMIC-IV clinical dataset.

Retrospectively collected medical data has the opportunity to improve patient care through knowledge discovery and
algorithm development. Broad reuse of medical data is desirable for the greatest public good, but data sharing must
be done in a manner which protects patient privacy. The Medical Information Mart for Intensive Care (MIMIC)-III
database provided critical care data for over 40,000 patients admitted to intensive care units at the
Beth Israel Deaconess Medical Center (BIDMC). Importantly, MIMIC-III was deidentified, and patient identifiers
were removed according to the Health Insurance Portability and Accountability Act (HIPAA) Safe Harbor provision.
MIMIC-III has been integral in driving large amounts of research in clinical informatics, epidemiology,
and machine learning. Here we present MIMIC-IV, an update to MIMIC-III, which incorporates contemporary data
and improves on numerous aspects of MIMIC-III. MIMIC-IV adopts a modular approach to data organization,
highlighting data provenance and facilitating both individual and combined use of disparate data sources.
MIMIC-IV is intended to carry on the success of MIMIC-III and support a broad set of applications within healthcare.
"""

__all__ = ["MIMIC_IV"]

from typing import Optional, Sequence

from tsdm.datasets import DATASET_OBJECT
from tsdm.datasets.base import MultiFrameDataset
from tsdm.utils.types import KeyVar, Nested, PathType


class MIMIC_IV(MultiFrameDataset):
    r"""MIMIC-IV clinical dataset."""

    BASE_URL: str = r"https://www.physionet.org/content/mimiciv/get-zip/1.0/"
    INFO_URL: str = r"https://www.physionet.org/content/mimiciv/1.0/"
    HOME_URL: str = r"https://mimic.mit.edu/"
    VERSION: str = "1.0"

    @property
    def index(self) -> Sequence[KeyVar]:
        r"""Return the index of the dataset."""
        raise NotImplementedError

    def _clean(self, key: KeyVar) -> DATASET_OBJECT | None:
        raise NotImplementedError

    @property
    def rawdata_files(self) -> Nested[Optional[PathType]]:
        r"""Return the raw data files."""
        raise NotImplementedError
