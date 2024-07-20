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

__all__ = ["MIMIC_III_RAW", "MIMIC_III"]

import gzip
from functools import cached_property
from getpass import getpass
from typing import get_args
from zipfile import ZipFile

from pandas import DataFrame
from pyarrow import Table, csv

from tsdm.backend.pyarrow import cast_columns, filter_nulls, set_nulls
from tsdm.data import strip_whitespace
from tsdm.datasets.base import MultiTableDataset
from tsdm.datasets.mimic.mimic_iii_schema import (
    FALSE_VALUES,
    KEYS,
    NULL_VALUES,
    SCHEMAS,
    TRUE_VALUES,
)


class MIMIC_III_RAW(MultiTableDataset[KEYS, DataFrame]):
    r"""Raw version of the MIMIC-III Clinical Database.

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

    Notes:
        NOTE: TIME_STAMP = round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36))
        and bin_k = 10
        i.e. TIME_STAMP = round(dt.total_seconds()*10/3600) = round(dt.total_hours()*10)
        i.e. TIME_STAMP ≈ 10*total_hours
        so e.g. the last patient was roughly 250 hours, 10½ days.
    """

    __version__ = "1.4"

    SOURCE_URL = r"https://physionet.org/content/mimiciii/get-zip/"
    INFO_URL = r"https://physionet.org/content/mimiciii/"
    HOME_URL = r"https://mimic.mit.edu/"

    rawdata_hashes = {
        "mimic-iii-clinical-database-1.4.zip": "sha256:f9917f0f77f29d9abeb4149c96724618923a4725310c62fb75529a2c3e483abd"
    }

    @cached_property
    def table_names(self) -> tuple[KEYS, ...]:
        names = tuple(self.filelist)
        if unknown_names := set(names) - set(get_args(KEYS)):
            raise ValueError(f"Unknown table names: {unknown_names!r}")
        return names

    @cached_property
    def rawdata_files(self) -> list[str]:
        return [f"mimic-iii-clinical-database-{self.__version__}.zip"]

    @cached_property
    def filelist(self) -> dict[KEYS, str]:
        r"""Mapping between table_names and contents of the zip file."""
        if not self.version_info >= (1, 4):
            raise ValueError("MIMIC-III v1.4+ is required.")

        return {
            key: f"mimic-iii-clinical-database-{self.__version__}/{key}.csv.gz"  # type: ignore[misc]
            for key in [
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
        }

    def clean_table(self, key: KEYS) -> Table:
        # Read the table
        with (
            ZipFile(self.rawdata_paths[self.rawdata_files[0]], "r") as archive,
            archive.open(self.filelist[key], "r") as compressed_file,
            gzip.open(compressed_file, "r") as file,
        ):
            table = csv.read_csv(
                file,
                convert_options=csv.ConvertOptions(
                    column_types=SCHEMAS[key],
                    strings_can_be_null=True,
                    null_values=NULL_VALUES,
                    true_values=TRUE_VALUES,
                    false_values=FALSE_VALUES,
                ),
                parse_options=csv.ParseOptions(
                    newlines_in_values=(key == "NOTEEVENTS"),
                ),
            )

        return table.combine_chunks()  # <- reduces size and avoids some bugs

    def download_file(self, fname: str, /) -> None:
        r"""Download a file from the MIMIC-III website."""
        if tuple(map(int, self.__version__.split("."))) < (1, 4):
            raise ValueError(
                "MIMIC-III v1.4+ is required. At the time of writing, the website"
                " does not provide legacy versions of the MIMIC-III dataset."
            )

        self.download_from_url(
            self.SOURCE_URL + f"{self.__version__}/",
            self.rawdata_paths[fname],
            username=input("MIMIC-III username: "),
            password=getpass(prompt="MIMIC-III password: ", stream=None),
            headers={
                "User-Agent": "Wget/1.21.2"
            },  # NOTE: MIMIC only allows wget for some reason...
        )


class MIMIC_III(MIMIC_III_RAW):
    r"""Lightly preprocessed version of the MIMIC-III dataset."""

    RAWDATA_DIR = MIMIC_III_RAW.RAWDATA_DIR

    def clean_table(self, key: KEYS) -> Table:
        table: Table = super().clean_table(key)

        # Post-processing
        match key:
            case "ADMISSIONS":
                table = set_nulls(
                    table,
                    ETHNICITY=["UNKNOWN/NOT SPECIFIED"],
                    RELIGION=["NOT SPECIFIED", "UNOBTAINABLE"],
                    MARTIAL_STATUS=["UNKNOWN (DEFAULT)"],
                )
            case "CALLOUT":
                pass
            case "CAREGIVERS":
                pass
            case "CHARTEVENTS":
                table = filter_nulls(
                    table, "ICUSTAY_ID", "VALUE", "VALUENUM", "VALUEUOM"
                )
                table = cast_columns(table, VALUE="float64")
            case "CPTEVENTS":
                table = cast_columns(table, CHARTDATE="date32")
            case "DATETIMEEVENTS":
                pass
            case "DIAGNOSES_ICD":
                pass
            case "DRGCODES":
                pass
            case "D_CPT":
                pass
            case "D_ICD_DIAGNOSES":
                pass
            case "D_ICD_PROCEDURES":
                pass
            case "D_ITEMS":
                pass
            case "D_LABITEMS":
                pass
            case "ICUSTAYS":
                pass
            case "INPUTEVENTS_CV":
                pass
            case "INPUTEVENTS_MV":
                pass
            case "LABEVENTS":
                table = filter_nulls(table, "VALUE", "VALUENUM", "VALUEUOM")
                table = strip_whitespace(table)
                table = cast_columns(table, VALUE="float64")
            case "MICROBIOLOGYEVENTS":
                table = cast_columns(table, CHARTDATE="date32")
            case "NOTEEVENTS":
                pass
            case "OUTPUTEVENTS":
                table = filter_nulls(table, "VALUE", "VALUEUOM")
                table = cast_columns(table, VALUE="float64")
            case "PATIENTS":
                table = cast_columns(
                    table,
                    DOB="date32",
                    DOD="date32",
                    DOD_HOSP="date32",
                    DOD_SSN="date32",
                )
            case "PRESCRIPTIONS":
                table = cast_columns(table, STARTDATE="date32", ENDDATE="date32")
            case "PROCEDUREEVENTS_MV":
                pass
            case "PROCEDURES_ICD":
                pass
            case "SERVICES":
                pass
            case "TRANSFERS":
                pass
            case _:
                raise ValueError(f"Unknown table name: {key}")

        return table
