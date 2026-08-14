r"""Custom processed version of the MIMIC-IV dataset."""

__all__ = [
    "BAD_NAN_COLUMNS",
    "BOOL_VALUES",
    "NULL_VALUES",
    "UNSTACKED_SCHEMAS",
    "MIMIC_IV_Scholz2026",
]

from typing import Any

import polars as pl

from tsdm.datasets.base import DatasetBase

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

UNSTACKED_SCHEMAS: dict[MIMIC_IV_Key, dict[Any, Any]] = {
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


def _strip_whitespace(table: pl.DataFrame, /, *columns: str) -> pl.DataFrame:
    r"""Strip whitespace from selected columns, or every string column by default."""
    expressions = (
        pl.col(*columns).str.strip_chars()
        if columns
        else pl.col(pl.String).str.strip_chars()
    )
    return table.with_columns(expressions)


def _assert_columns_equal(table: pl.DataFrame, /, *, left: str, right: str) -> None:
    r"""Raise an error unless two non-null columns contain identical values."""
    equal = table.select((pl.col(left) == pl.col(right)).all()).item()
    if not equal:
        raise AssertionError(f"{left} != {right}")


class MIMIC_IV_Scholz2026(DatasetBase[MIMIC_IV_Key, pl.DataFrame]):
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

    def clean_table(self, key: MIMIC_IV_Key) -> pl.DataFrame:
        dataset_path = self.raw_dataset.dataset_path[key]
        table = pl.read_parquet(dataset_path)

        # bool_values maps column names to their string-to-boolean values.
        if bool_values := BOOL_VALUES.get(key):
            table = table.with_columns(
                pl.col(column).replace_strict(values, return_dtype=BOOL_TYPE)
                for column, values in bool_values.items()
            )

        # drop data with missing `hadm_id`.
        if "hadm_id" in table.columns:
            table = table.filter(pl.col("hadm_id").is_not_null())

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
                table = _strip_whitespace(table, *cols).with_columns(
                    pl.col(column).cast(pl.Float32, strict=False) for column in cols
                )
            case "hcpcsevents":
                pass
            case "labevents":
                table = table.filter(
                    pl.col("storetime").is_not_null()
                    & pl.all_horizontal(
                        pl.col("value", "valuenum", "valueuom").is_not_null()
                    )
                )
                table = _strip_whitespace(table).with_columns(
                    pl.col("value").cast(pl.Float32)
                )
                _assert_columns_equal(table, left="value", right="valuenum")
                table = table.drop("valuenum")
            case "microbiologyevents":
                pass
            case "omr":
                # We pivot this table. This is complicated by the fact that the
                # value column contains both floats and tuples of floats of the form
                # (systolic, diastolic) for blood pressure measurements.
                table = table.with_columns(pl.col("result_value").str.split("/"))
                index = ["subject_id", "seq_num", "chartdate"]
                table = table.pivot(
                    on="result_name", index=index, values="result_value"
                )

                expressions: list[pl.Expr] = [pl.col(column) for column in index]
                for column in table.columns:
                    if column in index:
                        continue

                    values = pl.col(column)
                    match column:
                        case _ if "blood pressure" in column.lower():
                            expressions.extend(
                                (
                                    values.list.get(0, null_on_oob=True)
                                    .cast(pl.Float32)
                                    .alias(f"{column} (systolic)"),
                                    values.list.get(1, null_on_oob=True)
                                    .cast(pl.Float32)
                                    .alias(f"{column} (diastolic)"),
                                )
                            )
                        case "eGFR":
                            expressions.append(
                                values.list.get(0, null_on_oob=True)
                                .cast(pl.Utf8)
                                .alias(column)
                            )
                        case _:
                            expressions.append(
                                values.list.get(0, null_on_oob=True)
                                .cast(pl.Float32)
                                .alias(column)
                            )
                table = table.select(expressions).cast(UNSTACKED_SCHEMAS[key])
            case "patients":
                pass
            case "pharmacy":
                pass
            case "poe":
                pass
            case "poe_detail":
                table = table.pivot(
                    on="field_name",
                    index=["poe_id", "poe_seq", "subject_id"],
                    values="field_value",
                ).cast(UNSTACKED_SCHEMAS[key])
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
                table = table.filter(
                    pl.all_horizontal(
                        pl.col("value", "valuenum", "valueuom").is_not_null()
                    )
                ).with_columns(pl.col("value").cast(pl.Float64))
                _assert_columns_equal(table, left="value", right="valuenum")
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
                table = table.with_columns(
                    pl.col("storetime").dt.truncate("1s").cast(pl.Datetime("ms"))
                )
                seconds_per_unit = (
                    pl.col("valueuom")
                    .cast(pl.Utf8)
                    .replace_strict(
                        {"None": 0, "min": 60, "hour": 60 * 60, "day": 60 * 60 * 24},
                        return_dtype=pl.Int64,
                    )
                )
                duration = table.group_by("orderid").agg(
                    (pl.col("value").fill_null(0) * seconds_per_unit)
                    .sum()
                    .mul(1_000)
                    .round()
                    .cast(pl.Int64)
                    .cast(pl.Duration("ms"))
                    .alias("procedure_duration")
                )
                table = table.drop("value", "valueuom").join(
                    duration, on="orderid", how="left"
                )
            case _:
                raise KeyError(f"Unknown table name: {key}")

        return table
