r"""MIMIC-IV clinical dataset.

Abstract
--------
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


Preprocessing Details
---------------------
emar_detail
    1. cast/drop the following columns to float:
        - dose_due
        - dose_given
        - product_amount_given
        - prior_infusion_rate
        - infusion_rate
        - infusion_rate_adjustment_amount

labevents:
    1. drop data without hadm_id
    2. cast value to float

omr:
    1. convert from tall to wide by unstacking result_name/result_value
    2. split columns containing blood pressure into systolic and diastolic
    3. cast all generated columns to float

poe_detail
    1. Unstack on field_name/field_value

prescriptions:
    1. drop rows whose dose_val_rx/form_val_disp is not float.

procedureevents:
    1. convert storetime to second resolution

chartevents:
    1. Drop rows with missing valueuom
    2. cast value to float
    3. unstack value/valueuom??


Tables that may require unstacking
----------------------------------
- icu/chartevents
- icu/inputevents  <- has 3 of them...
- icu/outputevents
- icu/procedureevents
- icu/datetimeevents
- icu/ingredientevents
- hosp/labevents
- hosp/poe_detail   # <- unstack for sure
- hosp/pharmacy
- hosp/omr  # <- unstack for sure
"""

__all__ = ["MIMIC_IV"]


from collections.abc import Mapping
from getpass import getpass
from typing import Literal, TypeAlias, get_args

import pyarrow as pa
from pandas import DataFrame

from tsdm.datasets.base import MultiTableDataset

ID_TYPE = "uint32"
VALUE_TYPE = "float32"
TIME_TYPE = "timestamp[s]"
DATE_TYPE = "date32[day]"
BOOL_TYPE = "bool"
STRING_TYPE = "string"
DICT_TYPE = pa.dictionary("int32", "string")

TRUE_VALUES = ["Y", "Yes", "1"]
FALSE_VALUES = ["N", "No", "0"]
NULL_VALUES = [
    "        ",
    "       ",
    "      ",
    "     ",
    "    ",
    "   ",
    "  ",
    " ",
    "",
    "-",
    "---",
    "----",
    "-----",
    "-------",
    ".",
    ".*.",
    "?",
    "??",
    "???",
    "UNABLE TO OBTAIN",
    "UNKNOWN",
    "Unknown",
    "_",
    "__",
    "___",
    "___.",
    "unknown",
]


disallow_nan_values = {
    # fmt: off
    "admissions"         : ["admit_provider_id"],
    "d_hcpcs"            : (..., ["code", "short_description"]),
    "d_icd_diagnoses"    : [],
    "d_icd_procedures"   : [],
    "d_labitems"         : [],
    "diagnoses_icd"      : [],
    "drgcodes"           : ["drg_severity", "drg_mortality"],
    "emar"               : ["pharmacy_id", "enter_provider_id"],
    "emar_detail"        : ...,
    "hcpcsevents"        : [],
    "labevents"          : ["storetime"],
    "microbiologyevents" : ["storedate", "storetime", "spec_type_desc"],
    "omr"                : [],
    "patients"           : [],
    "pharmacy"           : [],
    "poe"                : ["order_provider_id"],
    "poe_detail"         : [],
    "prescriptions"      : [],
    "procedures_icd"     : [],
    "provider"           : [],
    "services"           : [],
    "transfers"          : [],
    "caregiver"          : [],
    "chartevents"        : ["valueuom"],
    "d_items"            : [],
    "datetimeevents"     : [],
    "icustays"           : [],
    "ingredientevents"   : [],
    "inputevents"        : [],
    "outputevents"       : [],
    "procedureevents"    : [],
    # fmt: on
}

KEYS: TypeAlias = Literal[
    "CHANGELOG",
    "LICENSE",
    "SHA256SUMS",
    "admissions",
    "d_hcpcs",
    "d_icd_diagnoses",
    "d_icd_procedures",
    "d_labitems",
    "diagnoses_icd",
    "drgcodes",
    "emar",
    "emar_detail",
    "hcpcsevents",
    "labevents",
    "microbiologyevents",
    "omr",
    "patients",
    "pharmacy",
    "poe",
    "poe_detail",
    "prescriptions",
    "procedures_icd",
    "provider",
    "services",
    "transfers",
    "caregiver",
    "chartevents",
    "d_items",
    "datetimeevents",
    "icustays",
    "ingredientevents",
    "inputevents",
    "outputevents",
    "procedureevents",
]


class MIMIC_IV(MultiTableDataset[KEYS, DataFrame]):
    r"""MIMIC-IV Clinical Database.

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

    BASE_URL = r"https://www.physionet.org/content/mimiciv/get-zip/"
    INFO_URL = r"https://www.physionet.org/content/mimiciv/"
    HOME_URL = r"https://mimic.mit.edu/"

    __version__ = "2.2"
    rawdata_hashes = {
        "mimic-iv-1.0.zip": "sha256:dd226e8694ad75149eed2840a813c24d5c82cac2218822bc35ef72e900baad3d",
        "mimic-iv-2.2.zip": "sha256:ddcedf49da4ff9a29ee25780b6ffc654d08af080fc1130dd0128a29514f21a74",
    }

    filelist: Mapping[KEYS, str] = {
        # fmt: off
        "CHANGELOG"  : "mimic-iv-2.2/CHANGELOG.txt",
        "LICENSE"    : "mimic-iv-2.2/LICENSE.txt",
        "SHA256SUMS" : "mimic-iv-2.2/SHA256SUMS.txt",
        "admissions"         : "mimic-iv-2.2/hosp/admissions.csv.gz",
        "d_hcpcs"            : "mimic-iv-2.2/hosp/d_hcpcs.csv.gz",
        "d_icd_diagnoses"    : "mimic-iv-2.2/hosp/d_icd_diagnoses.csv.gz",
        "d_icd_procedures"   : "mimic-iv-2.2/hosp/d_icd_procedures.csv.gz",
        "d_labitems"         : "mimic-iv-2.2/hosp/d_labitems.csv.gz",
        "diagnoses_icd"      : "mimic-iv-2.2/hosp/diagnoses_icd.csv.gz",
        "drgcodes"           : "mimic-iv-2.2/hosp/drgcodes.csv.gz",
        "emar"               : "mimic-iv-2.2/hosp/emar.csv.gz",
        "emar_detail"        : "mimic-iv-2.2/hosp/emar_detail.csv.gz",
        "hcpcsevents"        : "mimic-iv-2.2/hosp/hcpcsevents.csv.gz",
        "labevents"          : "mimic-iv-2.2/hosp/labevents.csv.gz",
        "microbiologyevents" : "mimic-iv-2.2/hosp/microbiologyevents.csv.gz",
        "omr"                : "mimic-iv-2.2/hosp/omr.csv.gz",
        "patients"           : "mimic-iv-2.2/hosp/patients.csv.gz",
        "pharmacy"           : "mimic-iv-2.2/hosp/pharmacy.csv.gz",
        "poe"                : "mimic-iv-2.2/hosp/poe.csv.gz",
        "poe_detail"         : "mimic-iv-2.2/hosp/poe_detail.csv.gz",
        "prescriptions"      : "mimic-iv-2.2/hosp/prescriptions.csv.gz",
        "procedures_icd"     : "mimic-iv-2.2/hosp/procedures_icd.csv.gz",
        "provider"           : "mimic-iv-2.2/hosp/provider.csv.gz",
        "services"           : "mimic-iv-2.2/hosp/services.csv.gz",
        "transfers"          : "mimic-iv-2.2/hosp/transfers.csv.gz",
        "caregiver"          : "mimic-iv-2.2/icu/caregiver.csv.gz",
        "chartevents"        : "mimic-iv-2.2/icu/chartevents.csv.gz",
        "d_items"            : "mimic-iv-2.2/icu/d_items.csv.gz",
        "datetimeevents"     : "mimic-iv-2.2/icu/datetimeevents.csv.gz",
        "icustays"           : "mimic-iv-2.2/icu/icustays.csv.gz",
        "ingredientevents"   : "mimic-iv-2.2/icu/ingredientevents.csv.gz",
        "inputevents"        : "mimic-iv-2.2/icu/inputevents.csv.gz",
        "outputevents"       : "mimic-iv-2.2/icu/outputevents.csv.gz",
        "procedureevents"    : "mimic-iv-2.2/icu/procedureevents.csv.gz",
        # fmt: on
    }
    table_names: tuple[KEYS, ...] = get_args(KEYS)

    @property
    def rawdata_files(self) -> list[str]:
        return [f"mimic-iv-{self.__version__}.zip"]

    def clean_table(self, key: str) -> None:
        ...

    def download_file(self, fname: str) -> None:
        self.download_from_url(
            self.BASE_URL + f"{self.__version__}/",
            self.rawdata_paths[fname],
            username=input("MIMIC-IV username: "),
            password=getpass(prompt="MIMIC-IV password: ", stream=None),
            headers={
                "User-Agent": "Wget/1.21.2"
            },  # NOTE: MIMIC only allows wget for some reason...
        )
