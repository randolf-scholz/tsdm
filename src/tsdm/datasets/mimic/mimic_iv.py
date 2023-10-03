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
1. emar_detail: cast/drop the following columns to float:
    - dose_due
    - dose_given
    - product_amount_given
    - prior_infusion_rate
    - infusion_rate
    - infusion_rate_adjustment_amount
2. labevents:
    - drop data without hadm_id
    - cast value to float
3. omr:
    - convert from tall to wide by unstacking result_name/result_value
    - split columns containing blood pressure into systolic and diastolic
    - cast all generated columns to float
4. poe_detail:  Unstack on field_name/field_value
5. prescriptions: drop rows whose dose_val_rx/form_val_disp is not float.
6. procedureevents: convert storetime to second resolution
7. chartevents:
    - Drop rows with missing valueuom
    - cast value to float
    - unstack value/valueuom??

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

__all__ = ["MIMIC_IV_RAW", "MIMIC_IV"]


import gzip
from getpass import getpass
from typing import get_args
from zipfile import ZipFile

import pandas as pd
import polars as pl
import pyarrow as pa
from pandas import DataFrame
from pyarrow import Array, Table, csv
from tqdm.autonotebook import tqdm

from tsdm.datasets.base import MultiTableDataset
from tsdm.datasets.mimic.mimic_iv_schema import (
    FALSE_VALUES,
    KEYS,
    NULL_VALUES,
    SCHEMAS,
    TRUE_VALUES,
    UNSTACKED_SCHEMAS,
)
from tsdm.utils.data import cast_columns, filter_nulls, force_cast, strip_whitespace

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


class MIMIC_IV_RAW(MultiTableDataset[KEYS, DataFrame]):
    r"""Raw version of the MIMIC-IV Clinical Database.

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
        "mimic-iv-1.0.zip": (
            "sha256:dd226e8694ad75149eed2840a813c24d5c82cac2218822bc35ef72e900baad3d"
        ),
        "mimic-iv-2.2.zip": (
            "sha256:ddcedf49da4ff9a29ee25780b6ffc654d08af080fc1130dd0128a29514f21a74"
        ),
    }

    table_names: tuple[KEYS, ...] = get_args(KEYS)

    @property
    def rawdata_files(self) -> list[str]:
        return [f"mimic-iv-{self.__version__}.zip"]

    @property
    def filelist(self) -> dict[KEYS, str]:
        """Mapping between table_names and contents of the zip file."""
        return {
            # fmt: off
            # "CHANGELOG"          : f"mimic-iv-{self.__version__}/CHANGELOG.txt",
            # "LICENSE"            : f"mimic-iv-{self.__version__}/LICENSE.txt",
            # "SHA256SUMS"         : f"mimic-iv-{self.__version__}/SHA256SUMS.txt",
            "admissions"         : f"mimic-iv-{self.__version__}/hosp/admissions.csv.gz",
            "d_hcpcs"            : f"mimic-iv-{self.__version__}/hosp/d_hcpcs.csv.gz",
            "d_icd_diagnoses"    : f"mimic-iv-{self.__version__}/hosp/d_icd_diagnoses.csv.gz",
            "d_icd_procedures"   : f"mimic-iv-{self.__version__}/hosp/d_icd_procedures.csv.gz",
            "d_labitems"         : f"mimic-iv-{self.__version__}/hosp/d_labitems.csv.gz",
            "diagnoses_icd"      : f"mimic-iv-{self.__version__}/hosp/diagnoses_icd.csv.gz",
            "drgcodes"           : f"mimic-iv-{self.__version__}/hosp/drgcodes.csv.gz",
            "emar"               : f"mimic-iv-{self.__version__}/hosp/emar.csv.gz",
            "emar_detail"        : f"mimic-iv-{self.__version__}/hosp/emar_detail.csv.gz",
            "hcpcsevents"        : f"mimic-iv-{self.__version__}/hosp/hcpcsevents.csv.gz",
            "labevents"          : f"mimic-iv-{self.__version__}/hosp/labevents.csv.gz",
            "microbiologyevents" : f"mimic-iv-{self.__version__}/hosp/microbiologyevents.csv.gz",
            "omr"                : f"mimic-iv-{self.__version__}/hosp/omr.csv.gz",
            "patients"           : f"mimic-iv-{self.__version__}/hosp/patients.csv.gz",
            "pharmacy"           : f"mimic-iv-{self.__version__}/hosp/pharmacy.csv.gz",
            "poe"                : f"mimic-iv-{self.__version__}/hosp/poe.csv.gz",
            "poe_detail"         : f"mimic-iv-{self.__version__}/hosp/poe_detail.csv.gz",
            "prescriptions"      : f"mimic-iv-{self.__version__}/hosp/prescriptions.csv.gz",
            "procedures_icd"     : f"mimic-iv-{self.__version__}/hosp/procedures_icd.csv.gz",
            "provider"           : f"mimic-iv-{self.__version__}/hosp/provider.csv.gz",
            "services"           : f"mimic-iv-{self.__version__}/hosp/services.csv.gz",
            "transfers"          : f"mimic-iv-{self.__version__}/hosp/transfers.csv.gz",
            "caregiver"          : f"mimic-iv-{self.__version__}/icu/caregiver.csv.gz",
            "chartevents"        : f"mimic-iv-{self.__version__}/icu/chartevents.csv.gz",
            "d_items"            : f"mimic-iv-{self.__version__}/icu/d_items.csv.gz",
            "datetimeevents"     : f"mimic-iv-{self.__version__}/icu/datetimeevents.csv.gz",
            "icustays"           : f"mimic-iv-{self.__version__}/icu/icustays.csv.gz",
            "ingredientevents"   : f"mimic-iv-{self.__version__}/icu/ingredientevents.csv.gz",
            "inputevents"        : f"mimic-iv-{self.__version__}/icu/inputevents.csv.gz",
            "outputevents"       : f"mimic-iv-{self.__version__}/icu/outputevents.csv.gz",
            "procedureevents"    : f"mimic-iv-{self.__version__}/icu/procedureevents.csv.gz",
            # fmt: on
        }

    def clean_table(self, key: KEYS) -> Table:
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
        self.download_from_url(
            self.BASE_URL + f"{self.__version__}/",
            self.rawdata_paths[fname],
            username=input("MIMIC-IV username: "),
            password=getpass(prompt="MIMIC-IV password: ", stream=None),
            headers={
                "User-Agent": "Wget/1.21.2"
            },  # NOTE: MIMIC only allows wget for some reason...
        )


class MIMIC_IV(MIMIC_IV_RAW):
    """Lightly preprocessed version of the MIMIC-IV dataset.

    The following preprocessing steps are applied:

    - data entries with missing hadm_id are dropped. affects:
        - hosp/(admissions, diagnoses_icd, drgcodes, emar, hcpcsevents, labevents,
          microbiologyevents, pharmacy, poe, prescriptions, procedures_icd, services, transfers
        - icu/chartevents, datetimeevents, icustays, ingredientevents, inputevents, outputevents, procedureevents
    - hosp/emar_detail:
        - trim whitespaces and cast the following columns to float, dropping incompatible values.
          dose_due, dose_given, product_amount_given, prior_infusion_rate,
          infusion_rate, infusion_rate_adjustment_amount,
    - hosp/labevents:
        - drop rows whose value/valuenum/valueuom is missing, convert value to float
        - drop rows whose storetime is missing
    - hosp/omr: unstack on value/valueuom
    - hosp/poe_detail: unstack on field_value/field_name
    - icu/chartevents: drop rows whose value/valuenum/valueuom is missing, convert value to float
    - icu/procedureevents:
        - storetime is converted to second resolution
        - unstack on value/valueuom, convert to new column "procedure_duration"
    """

    RAWDATA_DIR = MIMIC_IV_RAW.RAWDATA_DIR

    dataset_shapes = {
        # fmt: off
        "admissions"         : (  431231, 16),
        "d_hcpcs"            : (   89200,  4),
        "d_icd_diagnoses"    : (  109775,  3),
        "d_icd_procedures"   : (   85257,  3),
        "d_labitems"         : (    1622,  4),
        "diagnoses_icd"      : ( 4756326,  5),
        "drgcodes"           : (  604377,  7),
        "emar"               : (25035751, 12),
        "emar_detail"        : (54744789, 33),
        "hcpcsevents"        : (  150771,  6),
        "labevents"          : (47462620, 15),
        "microbiologyevents" : ( 1398317, 25),
        "omr"                : ( 2453539, 22),
        "patients"           : (  299712,  6),
        "pharmacy"           : (13584514, 27),
        "poe"                : (39366291, 12),
        "poe_detail"         : ( 2721856, 14),
        "prescriptions"      : (15416708, 21),
        "procedures_icd"     : (  669186,  6),
        "provider"           : (   40508,   ),
        "services"           : (  468029,  5),
        "transfers"          : ( 1560949,  7),
        "caregiver"          : (   15468,   ),
        "chartevents"        : (76942787, 10),
        "d_items"            : (    4014,  9),
        "datetimeevents"     : ( 7112999, 10),
        "icustays"           : (   73181,  8),
        "ingredientevents"   : (11627821, 17),
        "inputevents"        : ( 8978893, 26),
        "outputevents"       : ( 4234967,  9),
        "procedureevents"    : (  696092, 21),
        # fmt: on
    }

    def clean_table(self, key: KEYS) -> Table:
        table: Table = super().clean_table(key)

        # drop data with missing hadm_id
        if "hadm_id" in table.column_names:
            table = filter_nulls(table, "hadm_id")

        # post processing
        match key:
            case "admissions":
                pass
            case "d_hcpcs":
                pass
            case "d_icd_diagnoses":
                pass
            case "d_icd_procedures":
                pass
            case "d_labitems":
                pass
            case "diagnoses_icd":
                pass
            case "drgcodes":
                pass
            case "emar":
                pass
            case "emar_detail":
                cols = [
                    "dose_due",
                    "dose_given",
                    "product_amount_given",
                    "prior_infusion_rate",
                    "infusion_rate",
                    "infusion_rate_adjustment_amount",
                ]
                table = strip_whitespace(table, *cols)
                table = force_cast(table, **{col: pa.float32() for col in cols})
            case "hcpcsevents":
                pass
            case "labevents":
                table = filter_nulls(table, "storetime")
                table = filter_nulls(table, "value", "valuenum", "valueuom")
                table = cast_columns(table, value="float32")
                assert table["value"] == table["valuenum"]
                table = table.drop_columns("valuenum")
            case "microbiologyevents":
                pass
            case "omr":
                # We pivot this table. This is complicated by the fact that the
                # value column contains both floats and tuples of floats of the form
                # (systolic, diastolic) for blood pressure measurements.
                table = table.set_column(
                    table.column_names.index("result_value"),
                    "result_value",
                    pa.compute.split_pattern(table["result_value"], "/"),
                )

                # convert to pandas. Now each column contains NaN or list of floats.a
                df = table.to_pandas().pivot(
                    index=["subject_id", "seq_num", "chartdate"],
                    columns="result_name",
                    values="result_value",
                )

                for col in (pbar := tqdm(df.columns)):
                    pbar.set_postfix(column=f"{col!r}")

                    # Replace NaN with empty lists
                    s = df.pop(col).copy()
                    mask = s.isna()
                    s.loc[mask] = [[]] * mask.sum()  # list of empty lists

                    # blood pressure is a special case and results in 2 columns
                    columns = (
                        [f"{col} (systolic)", f"{col} (diastolic)"]
                        if "blood pressure" in col.lower()
                        else [col]
                    )
                    dtype = "float[pyarrow]" if col != "eGFR" else "string[pyarrow]"
                    frame = DataFrame(
                        s.to_list(), columns=columns, index=s.index, dtype=dtype
                    )
                    df[columns] = frame
                table = Table.from_pandas(df.reset_index())
                table = cast_columns(table, **UNSTACKED_SCHEMAS[key])
            case "patients":
                pass
            case "pharmacy":
                pass
            case "poe":
                pass
            case "poe_detail":
                table = (
                    pl.from_arrow(table)
                    .pivot(  # type: ignore[union-attr]
                        index=["poe_id", "poe_seq", "subject_id"],
                        columns="field_name",
                        values="field_value",
                    )
                    .to_arrow()
                )
                table = cast_columns(table, **UNSTACKED_SCHEMAS[key])
            case "prescriptions":
                pass
            case "procedures_icd":
                pass
            case "provider":
                pass
            case "services":
                pass
            case "transfers":
                pass
            case "caregiver":
                pass
            case "chartevents":
                table = filter_nulls(table, "value", "valuenum", "valueuom")
                table = cast_columns(table, value="float32")
                assert table["value"] == table["valuenum"]
                table = table.drop("valuenum")
            case "d_items":
                pass
            case "datetimeevents":
                pass
            case "icustays":
                pass
            case "ingredientevents":
                pass
            case "inputevents":
                pass
            case "outputevents":
                pass
            case "procedureevents":
                table = cast_columns(table, False, storetime="timestamp[s]")
                duration = (
                    table.to_pandas(types_mapper=pd.ArrowDtype)
                    .pivot(
                        index=["orderid"],
                        columns="valueuom",
                        values="value",
                    )
                    .fillna(0)
                    .dot(
                        pd.Series(
                            {
                                "None": 0,
                                "min": 60,
                                "day": 60 * 60 * 24,
                                "hour": 60 * 60,
                            },
                            dtype="duration[s][pyarrow]",
                            name="time",
                        )
                    )
                )
                table = table.set_column(
                    len(table.column_names),
                    "procedure_duration",
                    Array.from_pandas(duration, type="duration[s]", safe=False),
                )
                table = table.drop_columns(["value", "valueuom"])
            case _:
                raise ValueError(f"Unknown table name: {key}")

        return table