"""Schema for the KIWI-dataset."""

# NOTE: THis should only contain static data

__all__ = ["KEYS", "SCHEMAS", "TRUE_VALUES", "FALSE_VALUES", "NULL_VALUES"]

import pyarrow as pa
from typing_extensions import Literal, TypeAlias

ID_TYPE = pa.uint32()
VALUE_TYPE = pa.float64()
TIME_TYPE = pa.timestamp("s")
DATE_TYPE = pa.date32()
BOOL_TYPE = pa.bool_()
STRING_TYPE = pa.string()
DICT_TYPE = pa.dictionary(pa.int32(), pa.string())
NULL_TYPE = pa.null()
TEXT_TYPE = pa.large_utf8()

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

SCHEMAS: dict[KEYS, dict[str, pa.DataType]] = {
    "ADMISSIONS": {
        # fmt: off
        "ROW_ID"               : ID_TYPE,
        "SUBJECT_ID"           : ID_TYPE,
        "HADM_ID"              : ID_TYPE,
        "ADMITTIME"            : TIME_TYPE,
        "DISCHTIME"            : TIME_TYPE,
        "DEATHTIME"            : TIME_TYPE,
        "ADMISSION_TYPE"       : DICT_TYPE,
        "ADMISSION_LOCATION"   : DICT_TYPE,
        "DISCHARGE_LOCATION"   : DICT_TYPE,
        "INSURANCE"            : DICT_TYPE,
        "LANGUAGE"             : DICT_TYPE,
        "RELIGION"             : DICT_TYPE,
        "MARITAL_STATUS"       : DICT_TYPE,
        "ETHNICITY"            : DICT_TYPE,
        "EDREGTIME"            : TIME_TYPE,
        "EDOUTTIME"            : TIME_TYPE,
        "DIAGNOSIS"            : STRING_TYPE,
        "HOSPITAL_EXPIRE_FLAG" : BOOL_TYPE,
        "HAS_CHARTEVENTS_DATA" : BOOL_TYPE,
        # fmt: on
    },
    "CALLOUT": {
        # fmt: off
        "ROW_ID"                 : ID_TYPE,
        "SUBJECT_ID"             : ID_TYPE,
        "HADM_ID"                : ID_TYPE,
        "SUBMIT_WARDID"          : ID_TYPE,
        "SUBMIT_CAREUNIT"        : DICT_TYPE,
        "CURR_WARDID"            : ID_TYPE,
        "CURR_CAREUNIT"          : DICT_TYPE,
        "CALLOUT_WARDID"         : ID_TYPE,
        "CALLOUT_SERVICE"        : DICT_TYPE,
        "REQUEST_TELE"           : BOOL_TYPE,
        "REQUEST_RESP"           : BOOL_TYPE,
        "REQUEST_CDIFF"          : BOOL_TYPE,
        "REQUEST_MRSA"           : BOOL_TYPE,
        "REQUEST_VRE"            : BOOL_TYPE,
        "CALLOUT_STATUS"         : DICT_TYPE,
        "CALLOUT_OUTCOME"        : DICT_TYPE,
        "DISCHARGE_WARDID"       : ID_TYPE,
        "ACKNOWLEDGE_STATUS"     : DICT_TYPE,
        "CREATETIME"             : TIME_TYPE,
        "UPDATETIME"             : TIME_TYPE,
        "ACKNOWLEDGETIME"        : TIME_TYPE,
        "OUTCOMETIME"            : TIME_TYPE,
        "FIRSTRESERVATIONTIME"   : TIME_TYPE,
        "CURRENTRESERVATIONTIME" : TIME_TYPE,
        # fmt: on
    },
    "CAREGIVERS": {
        # fmt: off
        "ROW_ID"      : ID_TYPE,
        "CGID"        : ID_TYPE,
        "LABEL"       : DICT_TYPE,
        "DESCRIPTION" : DICT_TYPE,
        # fmt: on
    },
    "CHARTEVENTS": {
        # fmt: off
        "ROW_ID"       : ID_TYPE,
        "SUBJECT_ID"   : ID_TYPE,
        "HADM_ID"      : ID_TYPE,
        "ICUSTAY_ID"   : ID_TYPE,
        "ITEMID"       : ID_TYPE,
        "CHARTTIME"    : TIME_TYPE,
        "STORETIME"    : TIME_TYPE,
        "CGID"         : ID_TYPE,
        "VALUE"        : DICT_TYPE,  # FIXME: CAST FLOAT32
        "VALUENUM"     : VALUE_TYPE,  # FIXME: FILTER NULLS
        "VALUEUOM"     : DICT_TYPE,
        "WARNING"      : BOOL_TYPE,
        "ERROR"        : BOOL_TYPE,
        "RESULTSTATUS" : DICT_TYPE,
        "STOPPED"      : DICT_TYPE,
        # fmt: on
    },
    "CPTEVENTS": {
        # fmt: off
        "ROW_ID"           : ID_TYPE,
        "SUBJECT_ID"       : ID_TYPE,
        "HADM_ID"          : ID_TYPE,
        "COSTCENTER"       : DICT_TYPE,
        "CHARTDATE"        : TIME_TYPE,  # FIXME: cast date_type
        "CPT_CD"           : DICT_TYPE,
        "CPT_NUMBER"       : ID_TYPE,
        "CPT_SUFFIX"       : DICT_TYPE,
        "TICKET_ID_SEQ"    : ID_TYPE,
        "SECTIONHEADER"    : DICT_TYPE,
        "SUBSECTIONHEADER" : DICT_TYPE,
        "DESCRIPTION"      : DICT_TYPE,
        # fmt: on
    },
    "DATETIMEEVENTS": {
        # fmt: off
        "ROW_ID"       : ID_TYPE,
        "SUBJECT_ID"   : ID_TYPE,
        "HADM_ID"      : ID_TYPE,
        "ICUSTAY_ID"   : ID_TYPE,
        "ITEMID"       : ID_TYPE,
        "CHARTTIME"    : TIME_TYPE,
        "STORETIME"    : TIME_TYPE,
        "CGID"         : ID_TYPE,
        "VALUE"        : TIME_TYPE,
        "VALUEUOM"     : DICT_TYPE,
        "WARNING"      : BOOL_TYPE,
        "ERROR"        : BOOL_TYPE,
        "RESULTSTATUS" : DICT_TYPE,
        "STOPPED"      : DICT_TYPE,
        # fmt: on
    },
    "D_CPT": {
        # fmt: off
        "ROW_ID"              : ID_TYPE,
        "CATEGORY"            : ID_TYPE,
        "SECTIONRANGE"        : DICT_TYPE,
        "SECTIONHEADER"       : DICT_TYPE,
        "SUBSECTIONRANGE"     : STRING_TYPE,
        "SUBSECTIONHEADER"    : STRING_TYPE,
        "CODESUFFIX"          : BOOL_TYPE,
        "MINCODEINSUBSECTION" : ID_TYPE,
        "MAXCODEINSUBSECTION" : ID_TYPE,
        # fmt: on
    },
    "DIAGNOSES_ICD": {
        # fmt: off
        "ROW_ID"     : ID_TYPE,
        "SUBJECT_ID" : ID_TYPE,
        "HADM_ID"    : ID_TYPE,
        "SEQ_NUM"    : ID_TYPE,
        "ICD9_CODE"  : DICT_TYPE,
        # fmt: on
    },
    "D_ICD_DIAGNOSES": {
        # fmt: off
        "ROW_ID"      : ID_TYPE,
        "ICD9_CODE"   : STRING_TYPE,
        "SHORT_TITLE" : STRING_TYPE,
        "LONG_TITLE"  : STRING_TYPE,
        # fmt: on
    },
    "D_ICD_PROCEDURES": {
        # fmt: off
        "ROW_ID"      : ID_TYPE,
        "ICD9_CODE"   : STRING_TYPE,
        "SHORT_TITLE" : STRING_TYPE,
        "LONG_TITLE"  : STRING_TYPE,
        # fmt: on
    },
    "D_ITEMS": {
        # fmt: off
        "ROW_ID"       : ID_TYPE,
        "ITEMID"       : ID_TYPE,
        "LABEL"        : STRING_TYPE,
        "ABBREVIATION" : STRING_TYPE,
        "DBSOURCE"     : DICT_TYPE,
        "LINKSTO"      : DICT_TYPE,
        "CATEGORY"     : DICT_TYPE,
        "UNITNAME"     : DICT_TYPE,
        "PARAM_TYPE"   : DICT_TYPE,
        "CONCEPTID"    : NULL_TYPE,
        # fmt: on
    },
    "D_LABITEMS": {
        # fmt: off
        "ROW_ID"     : ID_TYPE,
        "ITEMID"     : ID_TYPE,
        "LABEL"      : STRING_TYPE,
        "FLUID"      : DICT_TYPE,
        "CATEGORY"   : DICT_TYPE,
        "LOINC_CODE" : STRING_TYPE,
        # fmt: on
    },
    "DRGCODES": {
        # fmt: off
        "ROW_ID"        : ID_TYPE,
        "SUBJECT_ID"    : ID_TYPE,
        "HADM_ID"       : ID_TYPE,
        "DRG_TYPE"      : DICT_TYPE,
        "DRG_CODE"      : ID_TYPE,
        "DESCRIPTION"   : DICT_TYPE,
        "DRG_SEVERITY"  : ID_TYPE,
        "DRG_MORTALITY" : ID_TYPE,
        # fmt: on
    },
    "ICUSTAYS": {
        # fmt: off
        "ROW_ID"         : ID_TYPE,
        "SUBJECT_ID"     : ID_TYPE,
        "HADM_ID"        : ID_TYPE,
        "ICUSTAY_ID"     : ID_TYPE,
        "DBSOURCE"       : DICT_TYPE,
        "FIRST_CAREUNIT" : DICT_TYPE,
        "LAST_CAREUNIT"  : DICT_TYPE,
        "FIRST_WARDID"   : ID_TYPE,
        "LAST_WARDID"    : ID_TYPE,
        "INTIME"         : TIME_TYPE,
        "OUTTIME"        : TIME_TYPE,
        "LOS"            : VALUE_TYPE,
        # fmt: on
    },
    "INPUTEVENTS_CV": {
        # fmt: off
        "ROW_ID"            : ID_TYPE,
        "SUBJECT_ID"        : ID_TYPE,
        "HADM_ID"           : ID_TYPE,
        "ICUSTAY_ID"        : ID_TYPE,
        "CHARTTIME"         : TIME_TYPE,
        "ITEMID"            : ID_TYPE,
        "AMOUNT"            : VALUE_TYPE,
        "AMOUNTUOM"         : DICT_TYPE,
        "RATE"              : VALUE_TYPE,
        "RATEUOM"           : DICT_TYPE,
        "STORETIME"         : TIME_TYPE,
        "CGID"              : ID_TYPE,
        "ORDERID"           : ID_TYPE,
        "LINKORDERID"       : ID_TYPE,
        "STOPPED"           : DICT_TYPE,
        "NEWBOTTLE"         : BOOL_TYPE,
        "ORIGINALAMOUNT"    : VALUE_TYPE,
        "ORIGINALAMOUNTUOM" : DICT_TYPE,
        "ORIGINALROUTE"     : DICT_TYPE,
        "ORIGINALRATE"      : VALUE_TYPE,
        "ORIGINALRATEUOM"   : DICT_TYPE,
        "ORIGINALSITE"      : DICT_TYPE,
        # fmt: on
    },
    "INPUTEVENTS_MV": {
        # fmt: off
        "ROW_ID"                        : ID_TYPE,
        "SUBJECT_ID"                    : ID_TYPE,
        "HADM_ID"                       : ID_TYPE,
        "ICUSTAY_ID"                    : ID_TYPE,
        "STARTTIME"                     : TIME_TYPE,
        "ENDTIME"                       : TIME_TYPE,
        "ITEMID"                        : ID_TYPE,
        "AMOUNT"                        : VALUE_TYPE,
        "AMOUNTUOM"                     : DICT_TYPE,
        "RATE"                          : VALUE_TYPE,
        "RATEUOM"                       : DICT_TYPE,
        "STORETIME"                     : TIME_TYPE,
        "CGID"                          : ID_TYPE,
        "ORDERID"                       : ID_TYPE,
        "LINKORDERID"                   : ID_TYPE,
        "ORDERCATEGORYNAME"             : DICT_TYPE,
        "SECONDARYORDERCATEGORYNAME"    : DICT_TYPE,
        "ORDERCOMPONENTTYPEDESCRIPTION" : DICT_TYPE,
        "ORDERCATEGORYDESCRIPTION"      : DICT_TYPE,
        "PATIENTWEIGHT"                 : VALUE_TYPE,
        "TOTALAMOUNT"                   : VALUE_TYPE,
        "TOTALAMOUNTUOM"                : DICT_TYPE,
        "ISOPENBAG"                     : BOOL_TYPE,
        "CONTINUEINNEXTDEPT"            : BOOL_TYPE,
        "CANCELREASON"                  : ID_TYPE,
        "STATUSDESCRIPTION"             : DICT_TYPE,
        "COMMENTS_EDITEDBY"             : DICT_TYPE,
        "COMMENTS_CANCELEDBY"           : DICT_TYPE,
        "COMMENTS_DATE"                 : TIME_TYPE,
        "ORIGINALAMOUNT"                : VALUE_TYPE,
        "ORIGINALRATE"                  : VALUE_TYPE,
        # fmt: on
    },
    "LABEVENTS": {
        # fmt: off
        "ROW_ID"     : ID_TYPE,
        "SUBJECT_ID" : ID_TYPE,
        "HADM_ID"    : ID_TYPE,
        "ITEMID"     : ID_TYPE,
        "CHARTTIME"  : TIME_TYPE,
        "VALUE"      : DICT_TYPE,  # FIXME: CAST FLOAT32
        "VALUENUM"   : VALUE_TYPE,  # FIXME: FILTER NULLS
        "VALUEUOM"   : DICT_TYPE,
        "FLAG"       : DICT_TYPE,
        # fmt: on
    },
    "MICROBIOLOGYEVENTS": {
        # fmt: off
        "ROW_ID"              : ID_TYPE,
        "SUBJECT_ID"          : ID_TYPE,
        "HADM_ID"             : ID_TYPE,
        "CHARTDATE"           : TIME_TYPE,  # FIXME: cast DATE
        "CHARTTIME"           : TIME_TYPE,
        "SPEC_ITEMID"         : ID_TYPE,
        "SPEC_TYPE_DESC"      : DICT_TYPE,
        "ORG_ITEMID"          : ID_TYPE,
        "ORG_NAME"            : DICT_TYPE,
        "ISOLATE_NUM"         : ID_TYPE,
        "AB_ITEMID"           : ID_TYPE,
        "AB_NAME"             : DICT_TYPE,
        "DILUTION_TEXT"       : DICT_TYPE,
        "DILUTION_COMPARISON" : DICT_TYPE,
        "DILUTION_VALUE"      : VALUE_TYPE,
        "INTERPRETATION"      : DICT_TYPE,
        # fmt: on
    },
    "NOTEEVENTS": {
        # fmt: off
        "ROW_ID"      : ID_TYPE,
        "SUBJECT_ID"  : ID_TYPE,
        "HADM_ID"     : ID_TYPE,
        "CHARTDATE"   : DATE_TYPE,
        "CHARTTIME"   : TIME_TYPE,
        "STORETIME"   : TIME_TYPE,
        "CATEGORY"    : DICT_TYPE,
        "DESCRIPTION" : DICT_TYPE,
        "CGID"        : ID_TYPE,
        "ISERROR"     : BOOL_TYPE,
        "TEXT"        : TEXT_TYPE,
        # fmt: on
    },
    "OUTPUTEVENTS": {
        # fmt: off
        "ROW_ID"     : ID_TYPE,
        "SUBJECT_ID" : ID_TYPE,
        "HADM_ID"    : ID_TYPE,
        "ICUSTAY_ID" : ID_TYPE,
        "CHARTTIME"  : TIME_TYPE,
        "ITEMID"     : ID_TYPE,
        "VALUE"      : VALUE_TYPE,  # FIXME: FILTER NULLS
        "VALUEUOM"   : DICT_TYPE,  # FIXME: FILTER NULLS
        "STORETIME"  : TIME_TYPE,
        "CGID"       : ID_TYPE,
        "STOPPED"    : NULL_TYPE,
        "NEWBOTTLE"  : NULL_TYPE,
        "ISERROR"    : NULL_TYPE,
        # fmt: on
    },
    "PATIENTS": {
        # fmt: off
        "ROW_ID"      : ID_TYPE,
        "SUBJECT_ID"  : ID_TYPE,
        "GENDER"      : DICT_TYPE,
        "DOB"         : TIME_TYPE,  # FIXME: cast DATE_TYPE
        "DOD"         : TIME_TYPE,  # FIXME: cast DATE_TYPE
        "DOD_HOSP"    : TIME_TYPE,  # FIXME: cast DATE_TYPE
        "DOD_SSN"     : TIME_TYPE,  # FIXME: cast DATE_TYPE
        "EXPIRE_FLAG" : BOOL_TYPE,
        # fmt: on
    },
    "PRESCRIPTIONS": {
        # fmt: off
        "ROW_ID"            : ID_TYPE,
        "SUBJECT_ID"        : ID_TYPE,
        "HADM_ID"           : ID_TYPE,
        "ICUSTAY_ID"        : ID_TYPE,
        "STARTDATE"         : TIME_TYPE,  # FIXME: cast DATE_TYPE
        "ENDDATE"           : TIME_TYPE,  # FIXME: cast DATE_TYPE
        "DRUG_TYPE"         : DICT_TYPE,
        "DRUG"              : DICT_TYPE,
        "DRUG_NAME_POE"     : DICT_TYPE,
        "DRUG_NAME_GENERIC" : DICT_TYPE,
        "FORMULARY_DRUG_CD" : DICT_TYPE,
        "GSN"               : DICT_TYPE,
        "NDC"               : DICT_TYPE,
        "PROD_STRENGTH"     : DICT_TYPE,
        "DOSE_VAL_RX"       : DICT_TYPE,
        "DOSE_UNIT_RX"      : DICT_TYPE,
        "FORM_VAL_DISP"     : DICT_TYPE,
        "FORM_UNIT_DISP"    : DICT_TYPE,
        "ROUTE"             : DICT_TYPE,
        # fmt: on
    },
    "PROCEDUREEVENTS_MV": {
        # fmt: off
        "ROW_ID"                     : ID_TYPE,
        "SUBJECT_ID"                 : ID_TYPE,
        "HADM_ID"                    : ID_TYPE,
        "ICUSTAY_ID"                 : ID_TYPE,
        "STARTTIME"                  : TIME_TYPE,
        "ENDTIME"                    : TIME_TYPE,
        "ITEMID"                     : ID_TYPE,
        "VALUE"                      : VALUE_TYPE,
        "VALUEUOM"                   : DICT_TYPE,
        "LOCATION"                   : DICT_TYPE,
        "LOCATIONCATEGORY"           : DICT_TYPE,
        "STORETIME"                  : TIME_TYPE,
        "CGID"                       : ID_TYPE,
        "ORDERID"                    : ID_TYPE,
        "LINKORDERID"                : ID_TYPE,
        "ORDERCATEGORYNAME"          : DICT_TYPE,
        "SECONDARYORDERCATEGORYNAME" : NULL_TYPE,
        "ORDERCATEGORYDESCRIPTION"   : DICT_TYPE,
        "ISOPENBAG"                  : BOOL_TYPE,
        "CONTINUEINNEXTDEPT"         : BOOL_TYPE,
        "CANCELREASON"               : ID_TYPE,
        "STATUSDESCRIPTION"          : DICT_TYPE,
        "COMMENTS_EDITEDBY"          : DICT_TYPE,
        "COMMENTS_CANCELEDBY"        : DICT_TYPE,
        "COMMENTS_DATE"              : TIME_TYPE,
        # fmt: on
    },
    "PROCEDURES_ICD": {
        # fmt: off
        "ROW_ID"     : ID_TYPE,
        "SUBJECT_ID" : ID_TYPE,
        "HADM_ID"    : ID_TYPE,
        "SEQ_NUM"    : ID_TYPE,
        "ICD9_CODE"  : DICT_TYPE,
        # fmt: on
    },
    "SERVICES": {
        # fmt: off
        "ROW_ID"       : ID_TYPE,
        "SUBJECT_ID"   : ID_TYPE,
        "HADM_ID"      : ID_TYPE,
        "TRANSFERTIME" : TIME_TYPE,
        "PREV_SERVICE" : DICT_TYPE,
        "CURR_SERVICE" : DICT_TYPE,
        # fmt: on
    },
    "TRANSFERS": {
        # fmt: off
        "ROW_ID"        : ID_TYPE,
        "SUBJECT_ID"    : ID_TYPE,
        "HADM_ID"       : ID_TYPE,
        "ICUSTAY_ID"    : ID_TYPE,
        "DBSOURCE"      : DICT_TYPE,
        "EVENTTYPE"     : DICT_TYPE,
        "PREV_CAREUNIT" : DICT_TYPE,
        "CURR_CAREUNIT" : DICT_TYPE,
        "PREV_WARDID"   : ID_TYPE,
        "CURR_WARDID"   : ID_TYPE,
        "INTIME"        : TIME_TYPE,
        "OUTTIME"       : TIME_TYPE,
        "LOS"           : VALUE_TYPE,
        # fmt: on
    },
}
