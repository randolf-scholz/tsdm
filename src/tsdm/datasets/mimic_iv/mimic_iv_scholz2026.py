r"""Custom processed version of the MIMIC-IV dataset."""

__all__ = [
    "BAD_NAN_COLUMNS",
    "BOOL_VALUES",
    "NULL_VALUES",
    "UNSTACKED_SCHEMAS",
    "MIMIC_IV_Scholz2026",
]

from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
from pyarrow import compute as pc
from tqdm.asyncio import tqdm

from tsdm.backend.pyarrow import (
    cast_columns,
    filter_nulls,
    force_cast,
    unsafe_cast_columns,
)
from tsdm.datatools import strip_whitespace

from .mimic_iv import (
    BOOL_TYPE,
    CAT_TYPE,
    DATE_TYPE,
    ID_TYPE,
    MIMIC_IV,
    STRING_TYPE,
    TIME_TYPE,
    VALUE_TYPE,
    MIMIC_IV_Key,
)

UNSTACKED_SCHEMAS: dict[MIMIC_IV_Key, dict[str, pa.DataType]] = {
    "omr": {
        "subject_id"                                   : ID_TYPE,
        "seq_num"                                      : ID_TYPE,
        "chartdate"                                    : DATE_TYPE,
        "Blood Pressure (systolic)"                    : VALUE_TYPE,
        "Blood Pressure (diastolic)"                   : VALUE_TYPE,
        "Weight (Lbs)"                                 : VALUE_TYPE,
        "BMI (kg/m2)"                                  : VALUE_TYPE,
        "Height (Inches)"                              : VALUE_TYPE,
        "Blood Pressure Sitting (systolic)"            : VALUE_TYPE,
        "Blood Pressure Sitting (diastolic)"           : VALUE_TYPE,
        "Blood Pressure Standing (1 min) (systolic)"   : VALUE_TYPE,
        "Blood Pressure Standing (1 min) (diastolic)"  : VALUE_TYPE,
        "Blood Pressure Lying (systolic)"              : VALUE_TYPE,
        "Blood Pressure Lying (diastolic)"             : VALUE_TYPE,
        "Blood Pressure Standing (3 mins) (systolic)"  : VALUE_TYPE,
        "Blood Pressure Standing (3 mins) (diastolic)" : VALUE_TYPE,
        "BMI"                                          : VALUE_TYPE,
        "Weight"                                       : VALUE_TYPE,
        "Blood Pressure Standing (systolic)"           : VALUE_TYPE,
        "Blood Pressure Standing (diastolic)"          : VALUE_TYPE,
        "eGFR"                                         : CAT_TYPE,
        "Height"                                       : VALUE_TYPE,
    },
    "poe_detail": {
        "poe_id"              : STRING_TYPE,
        "poe_seq"             : ID_TYPE,
        "subject_id"          : ID_TYPE,
        "Admit category"      : CAT_TYPE,
        "Admit to"            : CAT_TYPE,
        "Code status"         : CAT_TYPE,
        "Consult Status"      : CAT_TYPE,
        "Consult Status Time" : TIME_TYPE,
        "Discharge Planning"  : CAT_TYPE,
        "Discharge When"      : CAT_TYPE,
        "Indication"          : CAT_TYPE,
        "Level of Urgency"    : CAT_TYPE,
        "Transfer to"         : CAT_TYPE,
        "Tubes & Drains type" : CAT_TYPE,
    },
}  # fmt: skip
BOOL_VALUES: dict[str, dict[str, dict[Any, bool]]] = {
    "admissions": {"hospital_expire_flag": {0: False, 1: True}},
    "emar_detail": {
        "complete_dose_not_given": {"No": False, "Yes": True},
        "will_remainder_of_dose_be_given": {"No": False, "Yes": True},
        "infusion_complete": {"N": False, "Y": True},
        "new_iv_bag_hung": {"N": False, "Y": True},
        "continued_infusion_in_other_location": {"N": False, "Y": True},
        "non_formulary_visual_verification": {"N": False, "Y": True},
    },
    "pharmacy": {"sliding_scale": {"N": False, "Y": True}},
    "chartevents": {"warning": {0: False, 1: True}},
    "datetimeevents": {"warning": {0: False, 1: True}},
    "inputevents": {
        "isopenbag": {0: False, 1: True},
        "continueinnextdept": {0: False, 1: True},
    },
    "procedureevents": {
        "isopenbag": {0: False, 1: True},
        "continueinnextdept": {0: False, 1: True},
        "originalrate": {0: False, 1: True},
    },
}
BAD_NAN_COLUMNS = {
    "admissions"         : ["admit_provider_id"],
    "d_hcpcs"            : ["code", "short_description"],
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
}  # fmt: skip
NULL_VALUES = [
    "NaN",
    "NA",
    "*NEW*",
    "",
    " ",
    "  ",
    "   ",
    "    ",
    "     ",
    "      ",
    "       ",
    "        ",
    "-",
    "---",
    "----",
    "-----",
    "-------",
    "?",
    "UNABLE TO OBTAIN",
    "UNKNOWN",
    "Unknown",
    "unknown",
    ".",
    ".*.",
    "___.",
    "_",
    "__",
    "___",
    # "none",
    # "None",
    # "NONE",
]


class MIMIC_IV_Scholz2026(MIMIC_IV):
    r"""Lightly preprocessed version of the MIMIC-IV dataset.

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

    dataset_shapes = {
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
    }  # fmt: skip

    def __post_init__(self) -> None:
        # reuse the same data as the raw dataset
        self.raw_dataset = MIMIC_IV(version=self.version, initialize=False)
        self.RAWDATA_DIR = self.raw_dataset.RAWDATA_DIR

    def clean_table(self, key: MIMIC_IV_Key) -> pa.Table:
        dataset_path = self.raw_dataset.dataset_path[key]
        table = pl.scan_parquet(dataset_path)

        # bool_values maps column names to their string-to-boolean values.
        if bool_values := BOOL_VALUES.get(key):
            table = table.with_columns(
                pl.col(column).replace_strict(values, return_dtype=BOOL_TYPE)
                for column, values in bool_values.items()
            )

        # below: old pandas code!

        # drop data with missing `hadm_id`.
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
                table = strip_whitespace(table)
                table = cast_columns(table, value=pa.float32())
                if table["value"] != table["valuenum"]:
                    raise AssertionError("value != valuenum")
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
                    pc.split_pattern(table["result_value"], "/"),
                )

                # convert to pandas. Now each column contains NaN or list of floats.a
                df = table.to_pandas().pivot(
                    index=["subject_id", "seq_num", "chartdate"],
                    columns="result_name",
                    values="result_value",
                )

                for col in (pbar := tqdm(df.columns, desc="Fixing columns")):
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
                    frame = pd.DataFrame(
                        s.to_list(), columns=columns, index=s.index, dtype=dtype
                    )
                    df[columns] = frame
                table = pa.Table.from_pandas(df.reset_index())
                table = cast_columns(table, **UNSTACKED_SCHEMAS[key])
            case "patients":
                pass
            case "pharmacy":
                pass
            case "poe":
                pass
            case "poe_detail":
                # NOTE: we use polars because pandas is too slow.
                pl_frame = pl.from_arrow(table)
                assert isinstance(pl_frame, pl.DataFrame)
                table = pl_frame.pivot(
                    "field_name",
                    index=["poe_id", "poe_seq", "subject_id"],
                    values="field_value",
                ).to_arrow()
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
                table = cast_columns(table, value="float64")
                if table["value"] != table["valuenum"]:
                    raise AssertionError("value != valuenum")
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
                table = unsafe_cast_columns(table, storetime="timestamp[s]")
                time_conversion = pd.Series(
                    {"None": 0, "min": 60, "day": 60 * 60 * 24, "hour": 60 * 60},
                    dtype="duration[s][pyarrow]",
                    name="time",
                )
                duration = (
                    table.to_pandas(types_mapper=pd.ArrowDtype)
                    .pivot(
                        index=["orderid"],
                        columns="valueuom",  # "None", "min", "day", or "hour"
                        values="value",
                    )
                    .fillna(0)
                    .dot(time_conversion)
                )
                table = table.set_column(
                    len(table.column_names),  # <- append to end
                    "procedure_duration",
                    pa.Array.from_pandas(duration, type="duration[s]", safe=False),
                )
                table = table.drop_columns(["value", "valueuom"])
            case _:
                raise KeyError(f"Unknown table name: {key}")

        return table
