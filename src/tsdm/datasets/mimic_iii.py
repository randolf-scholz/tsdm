r"""MIMIC-III Clinical Database.

Abstract
--------
MIMIC-III is a large, freely-available database comprising de-identified health-related
data associated with over forty thousand patients who stayed in critical care units of
the Beth Israel Deaconess Medical Center between 2001 and 2012.
The database includes information such as demographics, vital sign measurements made at
the bedside (~1 data point per hour), laboratory test results, procedures, medications,
caregiver notes, imaging reports, and mortality (including post-hospital discharge).

MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical
decision-rule improvement, and electronic tool development. It is notable for three
factors: it is freely available to researchers worldwide; it encompasses a diverse and
very large population of ICU patients; and it contains highly granular data, including
vital signs, laboratory results, and medications.
"""

__all__ = ["MIMIC_III"]

from collections.abc import Mapping
from getpass import getpass
from typing import Literal, TypeAlias, get_args

from pandas import DataFrame

from tsdm.datasets.base import MultiTableDataset

KEYS: TypeAlias = Literal[
    "ADMISSIONS",
    "CALLOUT",
    "CAREGIVERS",
    "CHARTEVENTS",
    "CPTEVENTS",
    "DATETIMEEVENTS",
    "D_CPT",
    "DIAGNOSES_ICD",
    "D_ICD_DIAGNOSES",
    "D_ICD_PROCEDURES",
    "D_ITEMS",
    "D_LABITEMS",
    "DRGCODES",
    "ICUSTAYS",
    "INPUTEVENTS_CV",
    "INPUTEVENTS_MV",
    "LABEVENTS",
    "MICROBIOLOGYEVENTS",
    "NOTEEVENTS",
    "OUTPUTEVENTS",
    "PATIENTS",
    "PRESCRIPTIONS",
    "PROCEDUREEVENTS_MV",
    "PROCEDURES_ICD",
    "SERVICES",
    "TRANSFERS",
]


class MIMIC_III(MultiTableDataset[KEYS, DataFrame]):
    r"""MIMIC-III Clinical Database.

    MIMIC-III is a large, freely-available database comprising de-identified health-related data
    associated with over forty thousand patients who stayed in critical care units of the Beth
    Israel Deaconess Medical Center between 2001 and 2012. The database includes information such
    as demographics, vital sign measurements made at the bedside (~1 data point per hour),
    laboratory test results, procedures, medications, caregiver notes, imaging reports, and
    mortality (including post-hospital discharge).

    MIMIC supports a diverse range of analytic studies spanning epidemiology, clinical decision-rule
    improvement, and electronic tool development. It is notable for three factors: it is freely
    available to researchers worldwide; it encompasses a diverse and very large population of ICU
    patients; and it contains highly granular data, including vital signs, laboratory results,
    and medications.

    Notes
    -----
    NOTE: TIME_STAMP = round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36))
    and bin_k = 10
    i.e. TIME_STAMP = round(dt.total_seconds()*10/3600) = round(dt.total_hours()*10)
    i.e. TIME_STAMP ≈ 10*total_hours
    so e.g. the last patient was roughly 250 hours, 10½ days.
    """

    __version__ = "1.4"

    BASE_URL = r"https://physionet.org/content/mimiciii/get-zip/"
    INFO_URL = r"https://physionet.org/content/mimiciii/"
    HOME_URL = r"https://mimic.mit.edu/"

    rawdata_hashes = {
        "mimic-iii-clinical-database-1.4.zip": "sha256:f9917f0f77f29d9abeb4149c96724618923a4725310c62fb75529a2c3e483abd",  # noqa: E501
    }

    table_names: tuple[KEYS, ...] = get_args(KEYS)

    @property
    def rawdata_files(self):
        return [f"mimic-iii-clinical-database-{self.__version__}.zip"]

    @property
    def filelist(self) -> Mapping[KEYS, str]:
        """Mapping between table_names and contents of the zip file."""
        return {
            key: f"mimic-iii-clinical-database-{self.__version__}/{key}.csv.gz"
            for key in self.table_names
        }

    def clean_table(self, key: str) -> None:
        raise NotImplementedError

    def download_file(self, fname: str) -> None:
        """Download a file from the MIMIC-III website."""
        if tuple(map(int, self.__version__.split("."))) < (1, 4):
            raise ValueError(
                "MIMIC-III v1.4+ is required. At the time of writing, the website"
                " does not provide legacy versions of the MIMIC-III dataset."
            )

        self.download_from_url(
            self.BASE_URL + f"{self.__version__}/",
            self.rawdata_paths[fname],
            username=input("MIMIC-III username: "),
            password=getpass(prompt="MIMIC-III password: ", stream=None),
            headers={
                "User-Agent": "Wget/1.21.2"
            },  # NOTE: MIMIC only allows wget for some reason...
        )
