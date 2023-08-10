"""Schema for the KIWI-dataset."""

__all__ = ["KEYS", "SCHEMAS", "TRUE_VALUES", "FALSE_VALUES", "NULL_VALUES"]

from typing import Literal, TypeAlias

import pyarrow as pa

ID_TYPE = pa.uint32()
VALUE_TYPE = pa.float32()
TIME_TYPE = pa.timestamp("s")
DATE_TYPE = pa.date32()
BOOL_TYPE = pa.bool_()
STRING_TYPE = pa.string()
DICT_TYPE = pa.dictionary(pa.int32(), pa.string())
NULL_TYPE = pa.null()
TEXT_TYPE = pa.large_utf8()
INT8_TYPE = pa.int8()

# special values
TRUE_VALUES = ["Y", "Yes", "1", "T"]
FALSE_VALUES = ["N", "No", "0", "F"]
NULL_VALUES = [
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
]

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


SCHEMAS: dict[KEYS, dict[str, pa.DataType]] = {
    # NOTE: /ICU/ tables
    "chartevents": {
        # fmt: off
        "subject_id"   : ID_TYPE,
        "hadm_id"      : ID_TYPE,
        "stay_id"      : ID_TYPE,
        "caregiver_id" : ID_TYPE,
        "itemid"       : ID_TYPE,
        "charttime"    : TIME_TYPE,
        "storetime"    : TIME_TYPE,
        "value"        : STRING_TYPE,  # FIXME: cast float
        "valuenum"     : VALUE_TYPE,
        "valueuom"     : DICT_TYPE,
        "warning"      : BOOL_TYPE,
        # fmt: on
    },
    "inputevents": {
        # fmt: off
        "subject_id"                    : ID_TYPE,
        "hadm_id"                       : ID_TYPE,
        "stay_id"                       : ID_TYPE,
        "caregiver_id"                  : ID_TYPE,
        "starttime"                     : TIME_TYPE,
        "endtime"                       : TIME_TYPE,
        "storetime"                     : TIME_TYPE,
        "itemid"                        : ID_TYPE,
        "amount"                        : VALUE_TYPE,
        "amountuom"                     : DICT_TYPE,
        "rate"                          : VALUE_TYPE,
        "rateuom"                       : DICT_TYPE,
        "orderid"                       : ID_TYPE,
        "linkorderid"                   : ID_TYPE,
        "ordercategoryname"             : DICT_TYPE,
        "secondaryordercategoryname"    : DICT_TYPE,
        "ordercomponenttypedescription" : DICT_TYPE,
        "ordercategorydescription"      : DICT_TYPE,
        "patientweight"                 : VALUE_TYPE,
        "totalamount"                   : VALUE_TYPE,
        "totalamountuom"                : DICT_TYPE,
        "isopenbag"                     : BOOL_TYPE,
        "continueinnextdept"            : BOOL_TYPE,
        "statusdescription"             : DICT_TYPE,
        "originalamount"                : VALUE_TYPE,
        "originalrate"                  : VALUE_TYPE,
        # fmt: on
    },
    "outputevents": {
        # fmt: off
        "subject_id"  : ID_TYPE,
        "hadm_id"     : ID_TYPE,
        "stay_id"     : ID_TYPE,
        "caregiver_id": ID_TYPE,
        "charttime"   : TIME_TYPE,
        "storetime"   : TIME_TYPE,
        "itemid"      : ID_TYPE,
        "value"       : VALUE_TYPE,
        "valueuom"    : DICT_TYPE,
        # fmt: on
    },
    "procedureevents": {
        # fmt: off
        "subject_id"               : ID_TYPE,
        "hadm_id"                  : ID_TYPE,
        "stay_id"                  : ID_TYPE,
        "caregiver_id"             : ID_TYPE,
        "starttime"                : TIME_TYPE,
        "endtime"                  : TIME_TYPE,
        # "storetime"                : TIME_TYPE,
        "itemid"                   : ID_TYPE,
        "value"                    : VALUE_TYPE,
        "valueuom"                 : DICT_TYPE,
        "location"                 : DICT_TYPE,
        "locationcategory"         : DICT_TYPE,
        "orderid"                  : ID_TYPE,
        "linkorderid"              : ID_TYPE,
        "ordercategoryname"        : DICT_TYPE,
        "ordercategorydescription" : DICT_TYPE,
        "patientweight"            : VALUE_TYPE,
        "isopenbag"                : BOOL_TYPE,
        "continueinnextdept"       : BOOL_TYPE,
        "statusdescription"        : DICT_TYPE,
        "originalamount"           : VALUE_TYPE,
        "originalrate"             : BOOL_TYPE,
        # fmt: on
    },
    "datetimeevents": {
        # fmt: off
        "subject_id"   : ID_TYPE,
        "hadm_id"      : ID_TYPE,
        "stay_id"      : ID_TYPE,
        "caregiver_id" : ID_TYPE,
        "charttime"    : TIME_TYPE,
        "storetime"    : TIME_TYPE,
        "itemid"       : ID_TYPE,
        "value"        : TIME_TYPE,
        "valueuom"     : DICT_TYPE,
        "warning"      : BOOL_TYPE,
        # fmt: on
    },
    "ingredientevents": {
        # fmt: off
        "subject_id"        : ID_TYPE,
        "hadm_id"           : ID_TYPE,
        "stay_id"           : ID_TYPE,
        "caregiver_id"      : ID_TYPE,
        "starttime"         : TIME_TYPE,
        "endtime"           : TIME_TYPE,
        "storetime"         : TIME_TYPE,
        "itemid"            : ID_TYPE,
        "amount"            : VALUE_TYPE,
        "amountuom"         : DICT_TYPE,
        "rate"              : VALUE_TYPE,
        "rateuom"           : DICT_TYPE,
        "orderid"           : ID_TYPE,
        "linkorderid"       : ID_TYPE,
        "statusdescription" : DICT_TYPE,
        "originalamount"    : VALUE_TYPE,
        "originalrate"      : VALUE_TYPE,
        # fmt: on
    },
    "icustays": {
        # fmt: off
        "subject_id"     : ID_TYPE,
        "hadm_id"        : ID_TYPE,
        "stay_id"        : ID_TYPE,
        "first_careunit" : DICT_TYPE,
        "last_careunit"  : DICT_TYPE,
        "intime"         : TIME_TYPE,
        "outtime"        : TIME_TYPE,
        "los"            : VALUE_TYPE,
        # fmt: on
    },
    "d_items": {
        # fmt: off
        "itemid"          : ID_TYPE,
        "label"           : STRING_TYPE,
        "abbreviation"    : STRING_TYPE,
        "linksto"         : DICT_TYPE,
        "category"        : DICT_TYPE,
        "unitname"        : DICT_TYPE,
        "param_type"      : DICT_TYPE,
        "lownormalvalue"  : VALUE_TYPE,
        "highnormalvalue" : VALUE_TYPE,
        # fmt: on
    },
    "caregiver": {
        # fmt: off
        "caregiver_id": ID_TYPE,
        # fmt: on
    },
    # NOTE: /HOSP/ tables
    "admissions": {
        # fmt: off
        "subject_id"           : ID_TYPE,
        "hadm_id"              : ID_TYPE,
        "admittime"            : TIME_TYPE,
        "dischtime"            : TIME_TYPE,
        "deathtime"            : TIME_TYPE,
        "admission_type"       : DICT_TYPE,
        "admit_provider_id"    : DICT_TYPE,
        "admission_location"   : DICT_TYPE,
        "discharge_location"   : DICT_TYPE,
        "insurance"            : DICT_TYPE,
        "language"             : DICT_TYPE,
        "marital_status"       : DICT_TYPE,
        "race"                 : DICT_TYPE,
        "edregtime"            : TIME_TYPE,
        "edouttime"            : TIME_TYPE,
        "hospital_expire_flag" : BOOL_TYPE,
        # fmt: on
    },
    "d_hcpcs": {
        # fmt: off
        "code"              : STRING_TYPE,
        "category"          : INT8_TYPE,
        "long_description"  : TEXT_TYPE,
        "short_description" : DICT_TYPE,
        # fmt: on
    },
    "d_icd_diagnoses": {
        # fmt: off
        "icd_code"    : STRING_TYPE,
        "icd_version" : ID_TYPE,
        "long_title"  : STRING_TYPE,
        # fmt: on
    },
    "d_icd_procedures": {
        # fmt: off
        "icd_code"    : STRING_TYPE,
        "icd_version" : ID_TYPE,
        "long_title"  : STRING_TYPE,
        # fmt: on
    },
    "d_labitems": {
        # fmt: off
        "itemid"   : ID_TYPE,
        "label"    : STRING_TYPE,
        "fluid"    : DICT_TYPE,
        "category" : DICT_TYPE,
        # fmt: on
    },
    "diagnoses_icd": {
        # fmt: off
        "subject_id"  : ID_TYPE,
        "hadm_id"     : ID_TYPE,
        "seq_num"     : ID_TYPE,
        "icd_code"    : DICT_TYPE,
        "icd_version" : ID_TYPE,
        # fmt: on
    },
    "drgcodes": {
        # fmt: off
        "subject_id"    : ID_TYPE,
        "hadm_id"       : ID_TYPE,
        "drg_type"      : DICT_TYPE,
        "drg_code"      : DICT_TYPE,
        "description"   : DICT_TYPE,
        "drg_severity"  : INT8_TYPE,
        "drg_mortality" : INT8_TYPE,
        # fmt: on
    },
    "emar": {
        # fmt: off
        "subject_id"        : ID_TYPE,
        "hadm_id"           : ID_TYPE,  # FIXME: filter NULLS
        "emar_id"           : STRING_TYPE,
        "emar_seq"          : ID_TYPE,
        "poe_id"            : STRING_TYPE,
        "pharmacy_id"       : ID_TYPE,
        "enter_provider_id" : DICT_TYPE,
        "charttime"         : TIME_TYPE,
        "medication"        : DICT_TYPE,
        "event_txt"         : DICT_TYPE,
        "scheduletime"      : TIME_TYPE,
        "storetime"         : TIME_TYPE,
        # fmt: on
    },
    "emar_detail": {
        # fmt: off
        "subject_id"                           : ID_TYPE,
        "emar_id"                              : STRING_TYPE,
        "emar_seq"                             : ID_TYPE,
        "parent_field_ordinal"                 : DICT_TYPE,
        "administration_type"                  : DICT_TYPE,
        "pharmacy_id"                          : ID_TYPE,
        "barcode_type"                         : DICT_TYPE,
        "reason_for_no_barcode"                : DICT_TYPE,
        "complete_dose_not_given"              : BOOL_TYPE,
        "dose_due"                             : STRING_TYPE,  # FIXME: cast float
        "dose_due_unit"                        : DICT_TYPE,
        "dose_given"                           : STRING_TYPE,  # FIXME: cast float
        "dose_given_unit"                      : DICT_TYPE,
        "will_remainder_of_dose_be_given"      : BOOL_TYPE,
        "product_amount_given"                 : STRING_TYPE,  # FIXME: cast float
        "product_unit"                         : DICT_TYPE,
        "product_code"                         : DICT_TYPE,
        "product_description"                  : DICT_TYPE,
        "product_description_other"            : DICT_TYPE,
        "prior_infusion_rate"                  : STRING_TYPE,  # FIXME: cast float
        "infusion_rate"                        : STRING_TYPE,  # FIXME: cast float
        "infusion_rate_adjustment"             : DICT_TYPE,
        "infusion_rate_adjustment_amount"      : STRING_TYPE,  # FIXME: cast float
        "infusion_rate_unit"                   : DICT_TYPE,
        "route"                                : DICT_TYPE,
        "infusion_complete"                    : BOOL_TYPE,
        "completion_interval"                  : DICT_TYPE,
        "new_iv_bag_hung"                      : BOOL_TYPE,
        "continued_infusion_in_other_location" : BOOL_TYPE,
        "restart_interval"                     : DICT_TYPE,
        "side"                                 : DICT_TYPE,
        "site"                                 : DICT_TYPE,
        "non_formulary_visual_verification"    : BOOL_TYPE,
        # fmt: on
    },
    "hcpcsevents": {
        # fmt: off
        "subject_id"        : ID_TYPE,
        "hadm_id"           : ID_TYPE,
        "chartdate"         : DATE_TYPE,
        "hcpcs_cd"          : DICT_TYPE,
        "seq_num"           : ID_TYPE,
        "short_description" : DICT_TYPE,
        # fmt: on
    },
    "labevents": {
        # fmt: off
        "labevent_id"       : ID_TYPE,
        "subject_id"        : ID_TYPE,
        "hadm_id"           : ID_TYPE,
        "specimen_id"       : ID_TYPE,
        "itemid"            : ID_TYPE,
        "order_provider_id" : DICT_TYPE,
        "charttime"         : TIME_TYPE,
        "storetime"         : TIME_TYPE,
        "value"             : STRING_TYPE,  # FIXME: cast Float32
        "valuenum"          : VALUE_TYPE,
        "valueuom"          : DICT_TYPE,
        "ref_range_lower"   : VALUE_TYPE,
        "ref_range_upper"   : VALUE_TYPE,
        "flag"              : DICT_TYPE,
        "priority"          : DICT_TYPE,
        "comments"          : STRING_TYPE,
        # fmt: on
    },
    "microbiologyevents": {
        # fmt: off
        "microevent_id"       : ID_TYPE,
        "subject_id"          : ID_TYPE,
        "hadm_id"             : ID_TYPE,
        "micro_specimen_id"   : ID_TYPE,
        "order_provider_id"   : DICT_TYPE,
        "chartdate"           : TIME_TYPE,
        "charttime"           : TIME_TYPE,
        "spec_itemid"         : ID_TYPE,
        "spec_type_desc"      : DICT_TYPE,
        "test_seq"            : ID_TYPE,
        "storedate"           : TIME_TYPE,
        "storetime"           : TIME_TYPE,
        "test_itemid"         : ID_TYPE,
        "test_name"           : DICT_TYPE,
        "org_itemid"          : ID_TYPE,
        "org_name"            : DICT_TYPE,
        "isolate_num"         : ID_TYPE,
        "quantity"            : DICT_TYPE,
        "ab_itemid"           : ID_TYPE,
        "ab_name"             : DICT_TYPE,
        "dilution_text"       : STRING_TYPE,  # FIXME: convert to float
        "dilution_comparison" : DICT_TYPE,
        "dilution_value"      : VALUE_TYPE,
        "interpretation"      : DICT_TYPE,
        "comments"            : STRING_TYPE,
        # fmt: on
    },
    "omr": {
        # fmt: off
        "subject_id"   : ID_TYPE,
        "chartdate"    : DATE_TYPE,
        "seq_num"      : ID_TYPE,
        "result_name"  : STRING_TYPE,  # FIXME: unstack
        "result_value" : STRING_TYPE,
        # FIXME: split blood pressure into systolic/diastolic.
        # fmt: on
    },
    "patients": {
        # fmt: off
        "subject_id"        : ID_TYPE,
        "gender"            : DICT_TYPE,
        "anchor_age"        : ID_TYPE,
        "anchor_year"       : ID_TYPE,
        "anchor_year_group" : DICT_TYPE,
        "dod"               : DATE_TYPE,
        # fmt: on
    },
    "poe": {
        # fmt: off
        "poe_id"                 : STRING_TYPE,
        "poe_seq"                : ID_TYPE,
        "subject_id"             : ID_TYPE,
        "hadm_id"                : ID_TYPE,
        "ordertime"              : TIME_TYPE,
        "order_type"             : DICT_TYPE,
        "order_subtype"          : DICT_TYPE,
        "transaction_type"       : DICT_TYPE,
        "discontinue_of_poe_id"  : STRING_TYPE,
        "discontinued_by_poe_id" : STRING_TYPE,
        "order_provider_id"      : DICT_TYPE,
        "order_status"           : DICT_TYPE,
        # fmt: on
    },
    "poe_detail": {
        # fmt: off
        "poe_id"      : STRING_TYPE,
        "poe_seq"     : ID_TYPE,
        "subject_id"  : ID_TYPE,
        "field_name"  : DICT_TYPE,
        "field_value" : STRING_TYPE,
        # FIXME: unstack column
        # fmt: on
    },
    "prescriptions": {
        # fmt: off
        "subject_id"        : ID_TYPE,
        "hadm_id"           : ID_TYPE,
        "pharmacy_id"       : ID_TYPE,
        "poe_id"            : STRING_TYPE,
        "poe_seq"           : ID_TYPE,
        "order_provider_id" : DICT_TYPE,
        "starttime"         : TIME_TYPE,
        "stoptime"          : TIME_TYPE,
        "drug_type"         : DICT_TYPE,
        "drug"              : DICT_TYPE,
        "formulary_drug_cd" : DICT_TYPE,
        "gsn"               : DICT_TYPE,
        "ndc"               : DICT_TYPE,
        "prod_strength"     : DICT_TYPE,
        "form_rx"           : DICT_TYPE,
        "dose_val_rx"       : STRING_TYPE,  # FIXME: cast float
        "dose_unit_rx"      : DICT_TYPE,
        "form_val_disp"     : STRING_TYPE,  # FIXME: cast float
        "form_unit_disp"    : DICT_TYPE,
        "doses_per_24_hrs"  : VALUE_TYPE,
        "route"             : DICT_TYPE,
        # fmt: on
    },
    "pharmacy": {
        # fmt: off
        "subject_id"        :  ID_TYPE,
        "hadm_id"           :  ID_TYPE,
        "pharmacy_id"       :  ID_TYPE,
        "poe_id"            :  STRING_TYPE,
        "starttime"         :  TIME_TYPE,
        "stoptime"          :  TIME_TYPE,
        "medication"        :  DICT_TYPE,
        "proc_type"         :  DICT_TYPE,
        "status"            :  DICT_TYPE,
        "entertime"         :  TIME_TYPE,
        "verifiedtime"      :  TIME_TYPE,
        "route"             :  DICT_TYPE,
        "frequency"         :  DICT_TYPE,
        "disp_sched"        :  DICT_TYPE,
        "infusion_type"     :  DICT_TYPE,
        "sliding_scale"     :  BOOL_TYPE,
        "lockout_interval"  :  DICT_TYPE,
        "basal_rate"        :  VALUE_TYPE,
        "one_hr_max"        :  DICT_TYPE,
        "doses_per_24_hrs"  :  VALUE_TYPE,
        "duration"          :  VALUE_TYPE,
        "duration_interval" :  DICT_TYPE,
        "expiration_value"  :  VALUE_TYPE,
        "expiration_unit"   :  DICT_TYPE,
        "expirationdate"    :  TIME_TYPE,
        "dispensation"      :  DICT_TYPE,
        "fill_quantity"     :  DICT_TYPE,
        # fmt: on
    },
    "procedures_icd": {
        # fmt: off
        "subject_id"  : ID_TYPE,
        "hadm_id"     : ID_TYPE,
        "seq_num"     : ID_TYPE,
        "chartdate"   : DATE_TYPE,
        "icd_code"    : DICT_TYPE,
        "icd_version" : ID_TYPE,
        # fmt: on
    },
    "provider": {
        # fmt: off
        "provider_id": STRING_TYPE,
        # fmt: on
    },
    "services": {
        # fmt: off
        "subject_id"   : ID_TYPE,
        "hadm_id"      : ID_TYPE,
        "transfertime" : TIME_TYPE,
        "prev_service" : DICT_TYPE,
        "curr_service" : DICT_TYPE,
        # fmt: on
    },
    "transfers": {
        # fmt: off
        "subject_id"  : ID_TYPE,
        "hadm_id"     : ID_TYPE,
        "transfer_id" : ID_TYPE,
        "eventtype"   : DICT_TYPE,
        "careunit"    : DICT_TYPE,
        "intime"      : TIME_TYPE,
        "outtime"     : TIME_TYPE,
        # fmt: on
    },
}
