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
    - Cast values to float.
    - Unstack value/valueuom?

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

__all__ = [
    # Classes
    "MIMIC_IV_RAW",
    "MIMIC_IV",
    # Constants
    "MIMIC_IV_Key",
    "SCHEMAS",
    "NULL_VALUES",
    "BOOL_VALUES",
    "UNSTACKED_SCHEMAS",
    "BAD_NAN_COLUMNS",
    # Types
    "ID_TYPE",
    "VALUE_TYPE",
    "TIME_TYPE",
    "DATE_TYPE",
    "BOOL_TYPE",
    "STRING_TYPE",
    "CAT_TYPE",
    "NULL_TYPE",
    "TEXT_TYPE",
    "INT8_TYPE",
    # functions
    "rename_key",
    "insert_item",
]

from collections.abc import Mapping
from contextlib import suppress
from functools import cached_property
from getpass import getpass
from typing import Any, Literal, get_args
from zipfile import ZipFile

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from tqdm.auto import tqdm

from tsdm.backend.pyarrow import (
    cast_columns,
    filter_nulls,
    force_cast,
    unsafe_cast_columns,
)
from tsdm.datasets.base import DatasetBase
from tsdm.datatools import strip_whitespace, validate_schema
from tsdm.utils import remote

type MIMIC_IV_Key = Literal[
    # "CHANGELOG",
    # "LICENSE",
    # "SHA256SUMS",
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


def rename_key[K, V](items: dict[K, V], old_key: K, new_key: K, /) -> dict[K, V]:
    r"""Return a copy with ``old_key`` renamed to ``new_key`` in the same position."""
    if old_key not in items:
        raise KeyError(old_key)
    if new_key != old_key and new_key in items:
        raise KeyError(new_key)

    return {new_key if key == old_key else key: value for key, value in items.items()}


def insert_item[K, V](
    items: Mapping[K, V],
    key: K,
    value: V,
    /,
    *,
    before: K | None = None,
    after: K | None = None,
    at: int | None = None,
) -> dict[K, V]:
    r"""Return a copy with an item inserted at an explicit position.

    Exactly one of ``before``, ``after``, and ``at`` must be provided.
    """
    if sum(position is not None for position in (before, after, at)) != 1:
        raise ValueError("Specify exactly one of 'before', 'after', or 'at'.")
    if key in items:
        raise KeyError(key)

    keys = list(items)
    if at is not None:
        index = at
    elif before is not None:
        try:
            index = keys.index(before)
        except ValueError as error:
            raise KeyError(before) from error
    elif after is not None:
        try:
            index = keys.index(after) + 1
        except ValueError as error:
            raise KeyError(after) from error
    else:
        raise AssertionError("Invalid insertion position.")

    entries = list(items.items())
    entries.insert(index, (key, value))
    return dict(entries)


# region schema ------------------------------------------------------------------------


ID_TYPE = pl.UInt32
VALUE_TYPE = pl.Float32
TIME_TYPE = pl.Datetime("ms")
DATE_TYPE = pl.Date
BOOL_TYPE = pl.Boolean
STRING_TYPE = pl.Utf8
CAT_TYPE = pl.Categorical
NULL_TYPE = pl.Null
TEXT_TYPE = pl.Utf8
INT8_TYPE = pl.Int8


# based on version 1.0
SCHEMAS: dict[MIMIC_IV_Key, dict[str, pa.DataType]] = {
    # NOTE: /HOSP/ tables
    "admissions": {
        "subject_id"           : ID_TYPE,
        "hadm_id"              : ID_TYPE,
        "admittime"            : TIME_TYPE,
        "dischtime"            : TIME_TYPE,
        "deathtime"            : TIME_TYPE,
        "admission_type"       : CAT_TYPE,
        # "admit_provider_id"    : CAT_TYPE,  # added in v2.2
        "admission_location"   : CAT_TYPE,
        "discharge_location"   : CAT_TYPE,
        "insurance"            : CAT_TYPE,
        "language"             : CAT_TYPE,
        "marital_status"       : CAT_TYPE,
        "ethnicity"            : CAT_TYPE,  # renamed to race in v2.0
        "edregtime"            : TIME_TYPE,
        "edouttime"            : TIME_TYPE,
        "hospital_expire_flag" : INT8_TYPE,
    },
    "d_hcpcs": {
        "code"              : STRING_TYPE,
        "category"          : INT8_TYPE,
        "long_description"  : TEXT_TYPE,
        "short_description" : CAT_TYPE,
    },
    "d_icd_diagnoses": {
        "icd_code"    : CAT_TYPE,
        "icd_version" : CAT_TYPE,
        "long_title"  : STRING_TYPE,
    },
    "d_icd_procedures": {
        "icd_code"    : CAT_TYPE,
        "icd_version" : ID_TYPE,
        "long_title"  : STRING_TYPE,
    },
    "d_labitems": {
        "itemid"   : ID_TYPE,
        "label"    : STRING_TYPE,
        "fluid"    : CAT_TYPE,
        "category" : CAT_TYPE,
        "loinc_code" : STRING_TYPE,  # removed in v2.0
    },
    "diagnoses_icd": {
        "subject_id"  : ID_TYPE,
        "hadm_id"     : ID_TYPE,
        "seq_num"     : ID_TYPE,
        "icd_code"    : CAT_TYPE,
        "icd_version" : ID_TYPE,
    },
    "drgcodes": {
        "subject_id"    : ID_TYPE,
        "hadm_id"       : ID_TYPE,
        "drg_type"      : CAT_TYPE,
        "drg_code"      : CAT_TYPE,
        "description"   : CAT_TYPE,
        "drg_severity"  : INT8_TYPE,
        "drg_mortality" : INT8_TYPE,
    },
    "emar": {
        "subject_id"        : ID_TYPE,
        "hadm_id"           : ID_TYPE,  # NOTE: filter NULLS
        "emar_id"           : STRING_TYPE,  # range-like
        "emar_seq"          : ID_TYPE,
        "poe_id"            : STRING_TYPE,  # range-like
        "pharmacy_id"       : ID_TYPE,
        # "enter_provider_id" : CAT_TYPE,  # added in v2.2
        "charttime"         : TIME_TYPE,
        "medication"        : CAT_TYPE,
        "event_txt"         : CAT_TYPE,
        "scheduletime"      : TIME_TYPE,
        "storetime"         : TIME_TYPE,
    },
    "emar_detail": {
        "subject_id"                           : ID_TYPE,
        "emar_id"                              : STRING_TYPE,
        "emar_seq"                             : ID_TYPE,
        "parent_field_ordinal"                 : CAT_TYPE,
        "administration_type"                  : CAT_TYPE,
        "pharmacy_id"                          : ID_TYPE,
        "barcode_type"                         : CAT_TYPE,
        "reason_for_no_barcode"                : TEXT_TYPE,
        "complete_dose_not_given"              : CAT_TYPE,  # cast bool
        "dose_due"                             : STRING_TYPE,  # NOTE: cast float (range)
        "dose_due_unit"                        : CAT_TYPE,
        "dose_given"                           : STRING_TYPE,  # NOTE: cast float (range)
        "dose_given_unit"                      : CAT_TYPE,
        "will_remainder_of_dose_be_given"      : CAT_TYPE,  # cast bool
        "product_amount_given"                 : STRING_TYPE,  # NOTE: cast float (drop other)
        "product_unit"                         : CAT_TYPE,
        "product_code"                         : CAT_TYPE,
        "product_description"                  : CAT_TYPE,
        "product_description_other"            : CAT_TYPE,
        "prior_infusion_rate"                  : STRING_TYPE,  # NOTE: cast float (range)
        "infusion_rate"                        : STRING_TYPE,  # NOTE: cast float (range)
        "infusion_rate_adjustment"             : CAT_TYPE,
        "infusion_rate_adjustment_amount"      : STRING_TYPE,  # NOTE: cast float (drop other)
        "infusion_rate_unit"                   : CAT_TYPE,
        "route"                                : CAT_TYPE,
        "infusion_complete"                    : CAT_TYPE,  # cast bool
        "completion_interval"                  : CAT_TYPE,
        "new_iv_bag_hung"                      : CAT_TYPE,  # cast bool
        "continued_infusion_in_other_location" : CAT_TYPE,  # cast bool
        "restart_interval"                     : CAT_TYPE,
        "side"                                 : CAT_TYPE,
        "site"                                 : CAT_TYPE,
        "non_formulary_visual_verification"    : CAT_TYPE,  # cast bool
    },
    "hcpcsevents": {
        "subject_id"        : ID_TYPE,
        "hadm_id"           : ID_TYPE,
        "chartdate"         : DATE_TYPE,
        "hcpcs_cd"          : CAT_TYPE,
        "seq_num"           : ID_TYPE,
        "short_description" : CAT_TYPE,
    },
    "labevents": {
        "labevent_id"       : ID_TYPE,
        "subject_id"        : ID_TYPE,
        "hadm_id"           : ID_TYPE,  # NOTE: DROP MISSING ?
        "specimen_id"       : ID_TYPE,
        "itemid"            : ID_TYPE,
        # "order_provider_id" : STRING_ID_TYPE,  # added in v2.2
        "charttime"         : TIME_TYPE,
        "storetime"         : TIME_TYPE,
        "value"             : STRING_TYPE,
        "valuenum"          : VALUE_TYPE,  # NOTE: cast Float32
        "valueuom"          : CAT_TYPE,
        "ref_range_lower"   : VALUE_TYPE,
        "ref_range_upper"   : VALUE_TYPE,
        "flag"              : CAT_TYPE,
        "priority"          : CAT_TYPE,
        "comments"          : TEXT_TYPE,
    },
    "microbiologyevents": {
        "microevent_id"       : ID_TYPE,
        "subject_id"          : ID_TYPE,
        "hadm_id"             : ID_TYPE,
        "micro_specimen_id"   : ID_TYPE,
        # "order_provider_id"   : STRING_ID_TYPE,  # added in v2.2
        "chartdate"           : TIME_TYPE,
        "charttime"           : TIME_TYPE,
        "spec_itemid"         : ID_TYPE,
        "spec_type_desc"      : CAT_TYPE,
        "test_seq"            : ID_TYPE,
        "storedate"           : TIME_TYPE,
        "storetime"           : TIME_TYPE,
        "test_itemid"         : ID_TYPE,
        "test_name"           : CAT_TYPE,
        "org_itemid"          : ID_TYPE,
        "org_name"            : CAT_TYPE,
        "isolate_num"         : INT8_TYPE,
        "quantity"            : CAT_TYPE,
        "ab_itemid"           : ID_TYPE,
        "ab_name"             : CAT_TYPE,
        "dilution_text"       : CAT_TYPE,  # NOTE: comparison+value
        "dilution_comparison" : CAT_TYPE,
        "dilution_value"      : VALUE_TYPE,
        "interpretation"      : CAT_TYPE,
        "comments"            : TEXT_TYPE,
    },
    "omr": {
        "subject_id"   : ID_TYPE,
        "chartdate"    : DATE_TYPE,
        "seq_num"      : ID_TYPE,
        "result_name"  : CAT_TYPE,  # NOTE: unstack
        "result_value" : STRING_TYPE,  # NOTE: split blood pressure into systolic/diastolic.
    },
    "patients": {
        "subject_id"        : ID_TYPE,
        "gender"            : CAT_TYPE,
        "anchor_age"        : ID_TYPE,
        "anchor_year"       : ID_TYPE,
        "anchor_year_group" : CAT_TYPE,
        "dod"               : DATE_TYPE,
    },
    "pharmacy": {
        "subject_id"        :  ID_TYPE,
        "hadm_id"           :  ID_TYPE,
        "pharmacy_id"       :  ID_TYPE,
        "poe_id"            :  STRING_TYPE,
        "starttime"         :  TIME_TYPE,
        "stoptime"          :  TIME_TYPE,
        "medication"        :  TEXT_TYPE,
        "proc_type"         :  CAT_TYPE,
        "status"            :  CAT_TYPE,
        "entertime"         :  TIME_TYPE,
        "verifiedtime"      :  TIME_TYPE,
        "route"             :  CAT_TYPE,
        "frequency"         :  CAT_TYPE,
        "disp_sched"        :  CAT_TYPE,
        "infusion_type"     :  CAT_TYPE,
        "sliding_scale"     :  CAT_TYPE,  # convert to bool
        "lockout_interval"  :  CAT_TYPE,
        "basal_rate"        :  VALUE_TYPE,
        "one_hr_max"        :  CAT_TYPE,  # NOTE: cast float ??? (range)
        "doses_per_24_hrs"  :  VALUE_TYPE,
        "duration"          :  VALUE_TYPE,
        "duration_interval" :  CAT_TYPE,
        "expiration_value"  :  VALUE_TYPE,
        "expiration_unit"   :  CAT_TYPE,
        "expirationdate"    :  TIME_TYPE,
        "dispensation"      :  CAT_TYPE,
        "fill_quantity"     :  CAT_TYPE,
    },
    "poe": {
        "poe_id"                 : STRING_TYPE,
        "poe_seq"                : ID_TYPE,
        "subject_id"             : ID_TYPE,
        "hadm_id"                : ID_TYPE,
        "ordertime"              : TIME_TYPE,
        "order_type"             : CAT_TYPE,
        "order_subtype"          : CAT_TYPE,
        "transaction_type"       : CAT_TYPE,
        "discontinue_of_poe_id"  : STRING_TYPE,
        "discontinued_by_poe_id" : STRING_TYPE,
        # "order_provider_id"      : CAT_TYPE,  # added in v2.2
        "order_status"           : CAT_TYPE,
    },
    "poe_detail": {
        "poe_id"      : STRING_TYPE,
        "poe_seq"     : ID_TYPE,
        "subject_id"  : ID_TYPE,
        "field_name"  : CAT_TYPE,  # NOTE: unstack column
        "field_value" : STRING_TYPE,
    },
    "prescriptions": {
        "subject_id"        : ID_TYPE,
        "hadm_id"           : ID_TYPE,
        "pharmacy_id"       : ID_TYPE,
        # "order_provider_id" : CAT_TYPE,  # added in v2.2
        "starttime"         : TIME_TYPE,
        "stoptime"          : TIME_TYPE,
        "drug_type"         : CAT_TYPE,
        "drug"              : CAT_TYPE,
        # "formulary_drug_cd" : CAT_TYPE,  # added in v2.0
        "gsn"               : CAT_TYPE,
        "ndc"               : CAT_TYPE,
        "prod_strength"     : CAT_TYPE,
        "form_rx"           : CAT_TYPE,
        "dose_val_rx"       : STRING_TYPE,  # NOTE: cast float (range)
        "dose_unit_rx"      : CAT_TYPE,
        "form_val_disp"     : STRING_TYPE,  # NOTE: cast float (range)
        "form_unit_disp"    : CAT_TYPE,
        "doses_per_24_hrs"  : VALUE_TYPE,
        "route"             : CAT_TYPE,
    },
    "procedures_icd": {
        "subject_id"  : ID_TYPE,
        "hadm_id"     : ID_TYPE,
        "seq_num"     : ID_TYPE,
        "chartdate"   : DATE_TYPE,
        "icd_code"    : CAT_TYPE,
        "icd_version" : ID_TYPE,
    },
    "provider": {  # v2.2
        "provider_id" : STRING_TYPE,
    },
    "services": {
        "subject_id"   : ID_TYPE,
        "hadm_id"      : ID_TYPE,
        "transfertime" : TIME_TYPE,
        "prev_service" : CAT_TYPE,
        "curr_service" : CAT_TYPE,
    },
    "transfers": {
        "subject_id"  : ID_TYPE,
        "hadm_id"     : ID_TYPE,
        "transfer_id" : ID_TYPE,
        "eventtype"   : CAT_TYPE,
        "careunit"    : CAT_TYPE,
        "intime"      : TIME_TYPE,
        "outtime"     : TIME_TYPE,
    },
    # NOTE: /ICU/ tables
    "caregiver": {
        "caregiver_id": ID_TYPE,
    },
    "chartevents": {
        "subject_id"   : ID_TYPE,
        "hadm_id"      : ID_TYPE,
        "stay_id"      : ID_TYPE,
        # "caregiver_id" : ID_TYPE,  # added in v2.2
        "charttime"    : TIME_TYPE,
        "storetime"    : TIME_TYPE,
        "itemid"       : ID_TYPE,
        "value"        : STRING_TYPE,
        "valuenum"     : VALUE_TYPE,
        "valueuom"     : CAT_TYPE,
        "warning"      : INT8_TYPE,  # convert to bool
    },
    "d_items": {
        "itemid"          : ID_TYPE,
        "label"           : STRING_TYPE,
        "abbreviation"    : STRING_TYPE,
        "linksto"         : CAT_TYPE,
        "category"        : CAT_TYPE,
        "unitname"        : CAT_TYPE,
        "param_type"      : CAT_TYPE,
        "lownormalvalue"  : VALUE_TYPE,
        "highnormalvalue" : VALUE_TYPE,
    },
    "datetimeevents": {
        "subject_id"   : ID_TYPE,
        "hadm_id"      : ID_TYPE,
        "stay_id"      : ID_TYPE,
        # "caregiver_id" : ID_TYPE,
        "charttime"    : TIME_TYPE,
        "storetime"    : TIME_TYPE,
        "itemid"       : ID_TYPE,
        "value"        : TIME_TYPE,
        "valueuom"     : CAT_TYPE,  # NOTE: unstack?
        "warning"      : INT8_TYPE,  # convert to bool
    },
    "icustays": {
        "subject_id"     : ID_TYPE,
        "hadm_id"        : ID_TYPE,
        "stay_id"        : ID_TYPE,
        "first_careunit" : CAT_TYPE,
        "last_careunit"  : CAT_TYPE,
        "intime"         : TIME_TYPE,
        "outtime"        : TIME_TYPE,
        "los"            : VALUE_TYPE,
    },
    "ingredientevents": {
        "subject_id"        : ID_TYPE,
        "hadm_id"           : ID_TYPE,
        "stay_id"           : ID_TYPE,
        # "caregiver_id"      : ID_TYPE,  # added in v2.2
        "starttime"         : TIME_TYPE,
        "endtime"           : TIME_TYPE,
        "storetime"         : TIME_TYPE,
        "itemid"            : ID_TYPE,
        "amount"            : VALUE_TYPE,
        "amountuom"         : CAT_TYPE,
        "rate"              : VALUE_TYPE,
        "rateuom"           : CAT_TYPE,
        "orderid"           : ID_TYPE,
        "linkorderid"       : ID_TYPE,
        "statusdescription" : CAT_TYPE,
        "originalamount"    : VALUE_TYPE,
        "originalrate"      : VALUE_TYPE,
    },
    "inputevents": {
        "subject_id"                    : ID_TYPE,
        "hadm_id"                       : ID_TYPE,
        "stay_id"                       : ID_TYPE,
        # "caregiver_id"                  : ID_TYPE,  # added in v2.2
        "starttime"                     : TIME_TYPE,
        "endtime"                       : TIME_TYPE,
        "storetime"                     : TIME_TYPE,
        "itemid"                        : ID_TYPE,  # NOTE: unstack, but high-dim.
        "amount"                        : VALUE_TYPE,
        "amountuom"                     : CAT_TYPE,
        "rate"                          : VALUE_TYPE,
        "rateuom"                       : CAT_TYPE,
        "orderid"                       : ID_TYPE,
        "linkorderid"                   : ID_TYPE,
        "ordercategoryname"             : CAT_TYPE,
        "secondaryordercategoryname"    : CAT_TYPE,
        "ordercomponenttypedescription" : CAT_TYPE,
        "ordercategorydescription"      : CAT_TYPE,
        "patientweight"                 : VALUE_TYPE,
        "totalamount"                   : VALUE_TYPE,
        "totalamountuom"                : CAT_TYPE,
        "isopenbag"                     : INT8_TYPE,  # cast to bool
        "continueinnextdept"            : INT8_TYPE,  # cast to bool
        "cancelreason"                  : STRING_TYPE,  # removed in v2.2
        "statusdescription"             : CAT_TYPE,
        "originalamount"                : VALUE_TYPE,
        "originalrate"                  : VALUE_TYPE,
    },
    "outputevents": {
        "subject_id"   : ID_TYPE,
        "hadm_id"      : ID_TYPE,
        "stay_id"      : ID_TYPE,
        # "caregiver_id" : ID_TYPE,  # added in v2.2
        "charttime"    : TIME_TYPE,
        "storetime"    : TIME_TYPE,
        "itemid"       : ID_TYPE,
        "value"        : VALUE_TYPE,
        "valueuom"     : CAT_TYPE,
    },
    "procedureevents": {
        "subject_id"                 : ID_TYPE,
        "hadm_id"                    : ID_TYPE,
        "stay_id"                    : ID_TYPE,
        # "caregiver_id"               : ID_TYPE,  # added in v2.2
        "starttime"                  : TIME_TYPE,
        "endtime"                    : TIME_TYPE,
        "storetime"                  : TIME_TYPE,  # NOTE: cast to seconds
        "itemid"                     : ID_TYPE,
        "value"                      : VALUE_TYPE,  # NOTE: duration of procedure
        "valueuom"                   : CAT_TYPE,  # NOTE: unstack
        "location"                   : CAT_TYPE,
        "locationcategory"           : CAT_TYPE,
        "orderid"                    : ID_TYPE,
        "linkorderid"                : ID_TYPE,
        "ordercategoryname"          : CAT_TYPE,
        "secondaryordercategoryname" : STRING_TYPE,  # removed in v2.0
        "ordercategorydescription"   : CAT_TYPE,
        "patientweight"              : VALUE_TYPE,
        "totalamount"                : VALUE_TYPE,  # removed in v2.0
        "totalamountuom"             : CAT_TYPE,  # removed in v2.0
        "isopenbag"                  : INT8_TYPE,  # cast to bool
        "continueinnextdept"         : INT8_TYPE,  # cast to bool
        "cancelreason"               : STRING_TYPE,  # removed in v2.0
        "statusdescription"          : CAT_TYPE,
        "comments_date"              : TIME_TYPE,  # removed in v2.0
        "originalamount"             : VALUE_TYPE,
        "originalrate"               : INT8_TYPE,  # cast to bool
    },
}  # fmt: skip

# endregion schema ---------------------------------------------------------------------


class MIMIC_IV_RAW(DatasetBase[MIMIC_IV_Key, pl.LazyFrame]):
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


    ## CHANGELOG
    ## v3.1 (Oct 2024)

    Tables changes (data-only):
    `d_labitems`, `diagnoses_icd`, `drgcodes`, `labevents`, `microbiologyevents`, `omr`,
    `transfers`, `icustays`.

    ## v3.0 (Jul 23, 2024)

    Tables modified (data-only): `patients`, `admissions`, `icustays`.

    ## v2.2 (Jan 2023)

    Tables added: `caregiver`, `provider`.

    Tables modified:
    - `chartevents`, `datetimeevents`, `ingredientevents`, `inputevents`, `outputevents`,
      `procedureevents` (columns added: `caregiver_id`).
    - `admissions` (columns added: `admit_provider_id`).
    - `emar` (columns added: `enter_provider_id`).
    - `labevents`, `microbiologyevents`, `poe`, `prescriptions` (columns added: `order_provider_id`).

    ## v2.1 (Nov 16, 2022)

    Tables modified:
    - `patients`, `admissions`, `transfers`, `icustays` (removed some subject-ids)

    ## v2.0 (Jun 12, 2022)

    Tables added: `ingredientevents`, `omr`.

    Tables modified:
    - `admissions`, `patients`, `transfers` (schema change: moved from `core` module to `hosp` module).
    - `inputevents` (columns deleted: `cancelreason`).
    - `procedureevents` (columns deleted: `totalamount`, `totalamountuom`, `cancelreason`,
      `comments_editedby`, `comments_canceledby`, `comments_date`, `secondaryordercategoryname`).
    - `d_labitems` (columns deleted: `loinc_code`).
    - `prescriptions` (columns added: `formulary_drug_cd`).

    ## v1.0 (Mar 16, 2021)

    Tables modified:
    - `hcpcsevents` (columns added: `chartdate`).
    - `procedures_icd` (columns added: `chartdate`; columns modified/type: `icd_code` stored as `VARCHAR` (trimmed)).
    - `diagnoses_icd` (columns modified/type: `icd_code` stored as `VARCHAR` (trimmed)).
    """

    DEFAULT_VERSION = "1.0"
    __version__: str  # pyright: ignore[reportIncompatibleMethodOverride]

    SOURCE_URL = r"https://physionet.org/content/mimiciv/get-zip"
    CONTENT_URL = r"https://physionet.org/files/mimiciv"
    HOME_URL = r"https://mimic.mit.edu/"
    INFO_URL = r"https://physionet.org/content/mimiciv/"

    rawdata_hashes = {
        "mimic-iv-1.0.zip": "sha256:dd226e8694ad75149eed2840a813c24d5c82cac2218822bc35ef72e900baad3d",
        "mimic-iv-2.0.zip": "sha256:e11e9a56d234f2899714fb1712255abe0616dfcc6cba314178e8055b8765b3b9",
        "mimic-iv-2.2.zip": "sha256:ddcedf49da4ff9a29ee25780b6ffc654d08af080fc1130dd0128a29514f21a74",
    }

    @property
    def rawdata_files(self) -> list[str]:  # type: ignore
        return [f"mimic-iv-{self.__version__}.zip"]

    @property
    def table_names(self) -> list[MIMIC_IV_Key]:  # type: ignore
        expected_names = list(self.filelist)
        type_hinted_names = get_args(MIMIC_IV_Key.__value__)
        if unknown_names := set(expected_names) - set(type_hinted_names):
            raise ValueError(f"Unexpected table names: {unknown_names!r}")
        return expected_names

    @cached_property
    def filelist(self) -> dict[MIMIC_IV_Key, str]:
        r"""Mapping between table_names and contents of the zip file."""
        top = f"mimic-iv-{self.__version__}"

        files: dict[MIMIC_IV_Key, str] = {
            # "CHANGELOG"          : f"{top}/CHANGELOG.txt",
            # "LICENSE"            : f"{top}/LICENSE.txt",
            # "SHA256SUMS"         : f"{top}/SHA256SUMS.txt",
            # core
            "admissions"         : f"{top}/core/admissions.csv.gz",
            "patients"           : f"{top}/core/patients.csv.gz",
            "transfers"          : f"{top}/core/transfers.csv.gz",
            # hosp
            "d_hcpcs"            : f"{top}/hosp/d_hcpcs.csv.gz",
            "d_icd_diagnoses"    : f"{top}/hosp/d_icd_diagnoses.csv.gz",
            "d_icd_procedures"   : f"{top}/hosp/d_icd_procedures.csv.gz",
            "d_labitems"         : f"{top}/hosp/d_labitems.csv.gz",
            "diagnoses_icd"      : f"{top}/hosp/diagnoses_icd.csv.gz",
            "drgcodes"           : f"{top}/hosp/drgcodes.csv.gz",
            "emar"               : f"{top}/hosp/emar.csv.gz",
            "emar_detail"        : f"{top}/hosp/emar_detail.csv.gz",
            "hcpcsevents"        : f"{top}/hosp/hcpcsevents.csv.gz",
            "labevents"          : f"{top}/hosp/labevents.csv.gz",
            "microbiologyevents" : f"{top}/hosp/microbiologyevents.csv.gz",
            "pharmacy"           : f"{top}/hosp/pharmacy.csv.gz",
            # "omr"                : f"{top}/hosp/omr.csv.gz",              # NOTE: only version ≥2.0
            "poe"                : f"{top}/hosp/poe.csv.gz",
            "poe_detail"         : f"{top}/hosp/poe_detail.csv.gz",
            "prescriptions"      : f"{top}/hosp/prescriptions.csv.gz",
            "procedures_icd"     : f"{top}/hosp/procedures_icd.csv.gz",
            # "provider"           : f"{top}/hosp/provider.csv.gz",         # NOTE: only version ≥2.2
            "services"           : f"{top}/hosp/services.csv.gz",
            # icu
            # "caregiver"          : f"{top}/icu/caregiver.csv.gz",         # NOTE: only version ≥2.2
            "chartevents"        : f"{top}/icu/chartevents.csv.gz",
            "d_items"            : f"{top}/icu/d_items.csv.gz",
            "datetimeevents"     : f"{top}/icu/datetimeevents.csv.gz",
            "icustays"           : f"{top}/icu/icustays.csv.gz",
            "inputevents"        : f"{top}/icu/inputevents.csv.gz",
            # "ingredientevents"   : f"{top}/icu/ingredientevents.csv.gz",  # NOTE: only version ≥2.0
            "outputevents"       : f"{top}/icu/outputevents.csv.gz",
            "procedureevents"    : f"{top}/icu/procedureevents.csv.gz",
        }  # fmt: skip

        if self.version_info >= (2, 0):
            files |= {  # pyrefly: ignore[bad-assignment]
                "admissions"       : f"{top}/hosp/admissions.csv.gz",       # NOTE: changed folder
                "patients"         : f"{top}/hosp/patients.csv.gz",         # NOTE: changed folder
                "transfers"        : f"{top}/hosp/transfers.csv.gz",        # NOTE: changed folder
                "ingredientevents" : f"{top}/icu/ingredientevents.csv.gz",  # NOTE: new table
                "omr"              : f"{top}/hosp/omr.csv.gz",
            }  # fmt: skip

        if self.version_info >= (2, 2):
            files |= {  # pyrefly: ignore[bad-assignment]
                "caregiver"        : f"{top}/icu/caregiver.csv.gz",  # NOTE: new table
                "provider"         : f"{top}/hosp/provider.csv.gz",  # NOTE: new table
            }  # fmt: skip

        return files

    def get_schema(self, key) -> dict:
        schema = SCHEMAS[key].copy()
        if self.version_info >= (2, 0):
            match key:
                case "admissions":
                    # ethnicity got renamed to race in v2.0
                    schema = rename_key(schema, "ethnicity", "race")
                case "d_labitems":
                    del schema["loinc_code"]
                case "prescriptions":
                    schema = insert_item(
                        schema, "formulary_drug_cd", CAT_TYPE, after="drug"
                    )
                    schema = insert_item(
                        schema, "poe_id", STRING_TYPE, after="pharmacy_id"
                    )
                    schema = insert_item(schema, "poe_seq", STRING_TYPE, after="poe_id")
                case "inputevents":
                    del schema["cancelreason"]
                case "procedureevents":
                    del schema["totalamount"]
                    del schema["totalamountuom"]
                    del schema["cancelreason"]
                    del schema["comments_date"]
                    del schema["secondaryordercategoryname"]
                case _:
                    pass

        if self.version_info >= (2, 2):
            match key:
                case (
                    "chartevents"
                    | "datetimeevents"
                    | "ingredientevents"
                    | "inputevents"
                    | "outputevents"
                    | "procedureevents"
                ):
                    schema = insert_item(
                        schema, "caregiver_id", ID_TYPE, after="stay_id"
                    )
                case "admissions":
                    schema = insert_item(
                        schema, "admit_provider_id", CAT_TYPE, after="admission_type"
                    )
                case "emar":
                    schema = insert_item(
                        schema, "enter_provider_id", CAT_TYPE, after="pharmacy_id"
                    )
                case "labevents":
                    schema = insert_item(
                        schema, "order_provider_id", CAT_TYPE, after="itemid"
                    )
                case "microbiologyevents":
                    schema = insert_item(
                        schema,
                        "order_provider_id",
                        CAT_TYPE,
                        after="micro_specimen_id",
                    )
                case "poe":
                    schema = insert_item(
                        schema,
                        "order_provider_id",
                        CAT_TYPE,
                        after="discontinued_by_poe_id",
                    )
                case "prescriptions":
                    schema = insert_item(
                        schema, "order_provider_id", CAT_TYPE, after="poe_seq"
                    )
                case _:
                    pass

        return schema

    def clean_table(self, key: MIMIC_IV_Key) -> None:
        with (
            ZipFile(self.rawdata_paths[self.rawdata_files[0]], "r") as archive,
            archive.open(self.filelist[key], "r") as compressed_file,
        ):
            schema = self.get_schema(key)
            validate_schema(compressed_file, schema)
            table = pl.scan_csv(
                compressed_file,
                schema=schema,
            )

            try:
                table.sink_parquet(self.dataset_paths[key], engine="streaming")
            except BaseException:
                # delete partial file if an error occurs during writing
                with suppress(FileNotFoundError):
                    self.dataset_paths[key].unlink()
                raise

    def load_table(self, key: MIMIC_IV_Key, /) -> pl.LazyFrame:
        table = pl.scan_parquet(self.dataset_paths[key])
        table.collect_schema()
        return table

    def get_rawdata_file(self, fname: str, /) -> None:
        username = input("MIMIC-IV username: ")
        password = getpass(prompt="MIMIC-IV password: ", stream=None)

        if self.version_info == (2, 0):
            # zip file is not directly downloadable for 2.0
            remote.download_directory_to_zip(
                f"{self.CONTENT_URL}/{self.__version__}/",
                self.rawdata_paths[fname],
                username=username,
                password=password,
                headers={"User-Agent": "Wget/1.21.2"},
            )
        else:
            # direct zip available
            remote.download(
                f"{self.SOURCE_URL}/{self.__version__}/",
                self.rawdata_paths[fname],
                username=username,
                password=password,
                # NOTE: MIMIC only allows wget for some reason...
                headers={"User-Agent": "Wget/1.21.2"},
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


class MIMIC_IV(MIMIC_IV_RAW):
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
        self.raw_dataset = MIMIC_IV_RAW(version=self.__version__, initialize=False)
        self.RAWDATA_DIR = self.raw_dataset.RAWDATA_DIR
        super().__post_init__()

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

# special values
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
