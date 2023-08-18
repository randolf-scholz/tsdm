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
from getpass import getpass
from typing import get_args
from zipfile import ZipFile

from pandas import DataFrame
from pyarrow import Table, csv

from tsdm.datasets.base import MultiTableDataset
from tsdm.datasets.schema.mimic_iii import (
    FALSE_VALUES,
    KEYS,
    NULL_VALUES,
    SCHEMAS,
    TRUE_VALUES,
)
from tsdm.utils.data import cast_columns, filter_nulls, strip_whitespace


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
    def rawdata_files(self) -> list[str]:
        return [f"mimic-iii-clinical-database-{self.__version__}.zip"]

    @property
    def filelist(self) -> dict[KEYS, str]:
        """Mapping between table_names and contents of the zip file."""
        return {
            key: f"mimic-iii-clinical-database-{self.__version__}/{key}.csv.gz"
            for key in self.table_names
        }

    def clean_table(self, key: KEYS) -> Table:
        # Read the table
        with ZipFile(self.rawdata_paths[self.rawdata_files[0]], "r") as archive:
            with archive.open(self.filelist[key], "r") as compressed_file:
                with gzip.open(compressed_file, "r") as file:
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
                    ).combine_chunks()  # <- reduces size and avoids some bugs
                    # FIXME: https://github.com/apache/arrow/issues/37055

        return table

    def download_file(self, fname: str, /) -> None:
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


class MIMIC_III(MIMIC_III_RAW):
    """Lightly preprocessed version of the MIMIC-III dataset."""

    def clean_table(self, key: KEYS) -> Table:
        table: Table = super().clean_table(key)

        # Post-processing
        match key:
            case "ADMISSIONS":
                pass
            case "CALLOUT":
                pass
            case "CAREGIVERS":
                pass
            case "CHARTEVENTS":
                table = filter_nulls(
                    table, "ICUSTAY_ID", "VALUE", "VALUENUM", "VALUEUOM"
                )
                table = cast_columns(table, VALUE="float32")
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
                table = cast_columns(table, VALUE="float32")
            case "MICROBIOLOGYEVENTS":
                table = cast_columns(table, CHARTDATE="date32")
            case "NOTEEVENTS":
                pass
            case "OUTPUTEVENTS":
                table = filter_nulls(table, "VALUE", "VALUEUOM")
                table = cast_columns(table, VALUE="float32")
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
