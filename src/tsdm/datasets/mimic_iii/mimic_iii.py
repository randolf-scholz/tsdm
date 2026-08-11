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

__all__ = [
    # Classes
    "MIMIC_III",
    # Constants
    "MIMIC_III_Key",
    "SCHEMAS",
    # Types
    "ID_TYPE",
    "VALUE_TYPE",
    "TIME_TYPE",
    "DATE_TYPE",
    "BOOL_TYPE",
    "STRING_TYPE",
    "DICT_TYPE",
    "TEXT_TYPE",
    "INT8_TYPE",
]

from contextlib import suppress
from functools import cached_property
from getpass import getpass
from typing import Literal, get_args
from zipfile import ZipFile

import polars as pl
from polars.datatypes import DataType, DataTypeClass

from tsdm.datasets.base import DatasetBase
from tsdm.datatools import validate_schema
from tsdm.utils import remote

type MIMIC_III_Key = Literal[
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


ID_TYPE = pl.UInt32
VALUE_TYPE = pl.Float32
TIME_TYPE = pl.Datetime("ms")
DATE_TYPE = pl.Date
BOOL_TYPE = pl.Boolean
STRING_TYPE = pl.Utf8
DICT_TYPE = pl.Categorical
TEXT_TYPE = pl.Utf8
INT8_TYPE = pl.Int8


SCHEMAS: dict[MIMIC_III_Key, dict[str, DataType | DataTypeClass]] = {
    "ADMISSIONS": {
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
        "HOSPITAL_EXPIRE_FLAG" : INT8_TYPE,  # cast to bool
        "HAS_CHARTEVENTS_DATA" : INT8_TYPE,  # cast to bool
    },
    "CALLOUT": {
        "ROW_ID"                 : ID_TYPE,
        "SUBJECT_ID"             : ID_TYPE,
        "HADM_ID"                : ID_TYPE,
        "SUBMIT_WARDID"          : ID_TYPE,
        "SUBMIT_CAREUNIT"        : DICT_TYPE,
        "CURR_WARDID"            : ID_TYPE,
        "CURR_CAREUNIT"          : DICT_TYPE,
        "CALLOUT_WARDID"         : ID_TYPE,
        "CALLOUT_SERVICE"        : DICT_TYPE,
        "REQUEST_TELE"           : INT8_TYPE,  # cast to bool
        "REQUEST_RESP"           : INT8_TYPE,  # cast to bool
        "REQUEST_CDIFF"          : INT8_TYPE,  # cast to bool
        "REQUEST_MRSA"           : INT8_TYPE,  # cast to bool
        "REQUEST_VRE"            : INT8_TYPE,  # cast to bool
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
    },
    "CAREGIVERS": {
        "ROW_ID"      : ID_TYPE,
        "CGID"        : ID_TYPE,
        "LABEL"       : DICT_TYPE,
        "DESCRIPTION" : DICT_TYPE,
    },
    "CHARTEVENTS": {
        "ROW_ID"       : ID_TYPE,
        "SUBJECT_ID"   : ID_TYPE,
        "HADM_ID"      : ID_TYPE,
        "ICUSTAY_ID"   : ID_TYPE,
        "ITEMID"       : ID_TYPE,
        "CHARTTIME"    : TIME_TYPE,
        "STORETIME"    : TIME_TYPE,
        "CGID"         : ID_TYPE,
        "VALUE"        : STRING_TYPE,  # FIXME: CAST FLOAT32
        "VALUENUM"     : VALUE_TYPE,  # FIXME: FILTER NULLS
        "VALUEUOM"     : DICT_TYPE,
        "WARNING"      : INT8_TYPE,  # cast to bool
        "ERROR"        : INT8_TYPE,  # cast to bool
        "RESULTSTATUS" : DICT_TYPE,
        "STOPPED"      : DICT_TYPE,
    },
    "CPTEVENTS": {
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
    },
    "DATETIMEEVENTS": {
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
        "WARNING"      : INT8_TYPE,  # cast to bool
        "ERROR"        : INT8_TYPE,  # cast to bool
        "RESULTSTATUS" : DICT_TYPE,
        "STOPPED"      : DICT_TYPE,
    },
    "D_CPT": {
        "ROW_ID"              : ID_TYPE,
        "CATEGORY"            : ID_TYPE,
        "SECTIONRANGE"        : DICT_TYPE,
        "SECTIONHEADER"       : DICT_TYPE,
        "SUBSECTIONRANGE"     : STRING_TYPE,
        "SUBSECTIONHEADER"    : STRING_TYPE,
        "CODESUFFIX"          : DICT_TYPE,  # cast to bool
        "MINCODEINSUBSECTION" : ID_TYPE,
        "MAXCODEINSUBSECTION" : ID_TYPE,
    },
    "DIAGNOSES_ICD": {
        "ROW_ID"     : ID_TYPE,
        "SUBJECT_ID" : ID_TYPE,
        "HADM_ID"    : ID_TYPE,
        "SEQ_NUM"    : ID_TYPE,
        "ICD9_CODE"  : DICT_TYPE,
    },
    "D_ICD_DIAGNOSES": {
        "ROW_ID"      : ID_TYPE,
        "ICD9_CODE"   : STRING_TYPE,
        "SHORT_TITLE" : STRING_TYPE,
        "LONG_TITLE"  : STRING_TYPE,
    },
    "D_ICD_PROCEDURES": {
        "ROW_ID"      : ID_TYPE,
        "ICD9_CODE"   : STRING_TYPE,
        "SHORT_TITLE" : STRING_TYPE,
        "LONG_TITLE"  : STRING_TYPE,
    },
    "D_ITEMS": {
        "ROW_ID"       : ID_TYPE,
        "ITEMID"       : ID_TYPE,
        "LABEL"        : STRING_TYPE,
        "ABBREVIATION" : STRING_TYPE,
        "DBSOURCE"     : DICT_TYPE,
        "LINKSTO"      : DICT_TYPE,
        "CATEGORY"     : DICT_TYPE,
        "UNITNAME"     : DICT_TYPE,
        "PARAM_TYPE"   : DICT_TYPE,
        "CONCEPTID"    : STRING_TYPE,  # All Null
    },
    "D_LABITEMS": {
        "ROW_ID"     : ID_TYPE,
        "ITEMID"     : ID_TYPE,
        "LABEL"      : STRING_TYPE,
        "FLUID"      : DICT_TYPE,
        "CATEGORY"   : DICT_TYPE,
        "LOINC_CODE" : STRING_TYPE,
    },
    "DRGCODES": {
        "ROW_ID"        : ID_TYPE,
        "SUBJECT_ID"    : ID_TYPE,
        "HADM_ID"       : ID_TYPE,
        "DRG_TYPE"      : DICT_TYPE,
        "DRG_CODE"      : ID_TYPE,
        "DESCRIPTION"   : DICT_TYPE,
        "DRG_SEVERITY"  : ID_TYPE,
        "DRG_MORTALITY" : ID_TYPE,
    },
    "ICUSTAYS": {
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
    },
    "INPUTEVENTS_CV": {
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
        "NEWBOTTLE"         : INT8_TYPE,  # cast to bool
        "ORIGINALAMOUNT"    : VALUE_TYPE,
        "ORIGINALAMOUNTUOM" : DICT_TYPE,
        "ORIGINALROUTE"     : DICT_TYPE,
        "ORIGINALRATE"      : VALUE_TYPE,
        "ORIGINALRATEUOM"   : DICT_TYPE,
        "ORIGINALSITE"      : DICT_TYPE,
    },
    "INPUTEVENTS_MV": {
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
        "ISOPENBAG"                     : INT8_TYPE,  # cast to bool
        "CONTINUEINNEXTDEPT"            : INT8_TYPE,  # cast to bool
        "CANCELREASON"                  : ID_TYPE,
        "STATUSDESCRIPTION"             : DICT_TYPE,
        "COMMENTS_EDITEDBY"             : DICT_TYPE,
        "COMMENTS_CANCELEDBY"           : DICT_TYPE,
        "COMMENTS_DATE"                 : TIME_TYPE,
        "ORIGINALAMOUNT"                : VALUE_TYPE,
        "ORIGINALRATE"                  : VALUE_TYPE,
    },
    "LABEVENTS": {
        "ROW_ID"     : ID_TYPE,
        "SUBJECT_ID" : ID_TYPE,
        "HADM_ID"    : ID_TYPE,
        "ITEMID"     : ID_TYPE,
        "CHARTTIME"  : TIME_TYPE,
        "VALUE"      : DICT_TYPE,  # FIXME: CAST FLOAT32
        "VALUENUM"   : VALUE_TYPE,  # FIXME: FILTER NULLS
        "VALUEUOM"   : DICT_TYPE,
        "FLAG"       : DICT_TYPE,
    },
    "MICROBIOLOGYEVENTS": {
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
    },
    "NOTEEVENTS": {
        "ROW_ID"      : ID_TYPE,
        "SUBJECT_ID"  : ID_TYPE,
        "HADM_ID"     : ID_TYPE,
        "CHARTDATE"   : DATE_TYPE,
        "CHARTTIME"   : TIME_TYPE,
        "STORETIME"   : TIME_TYPE,
        "CATEGORY"    : DICT_TYPE,
        "DESCRIPTION" : DICT_TYPE,
        "CGID"        : ID_TYPE,
        "ISERROR"     : INT8_TYPE,  # cast to bool
        "TEXT"        : TEXT_TYPE,
    },
    "OUTPUTEVENTS": {
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
        "STOPPED"    : STRING_TYPE,  # All Null
        "NEWBOTTLE"  : STRING_TYPE,  # All Null
        "ISERROR"    : STRING_TYPE,  # All Null
    },
    "PATIENTS": {
        "ROW_ID"      : ID_TYPE,
        "SUBJECT_ID"  : ID_TYPE,
        "GENDER"      : DICT_TYPE,
        "DOB"         : TIME_TYPE,  # FIXME: cast DATE_TYPE
        "DOD"         : TIME_TYPE,  # FIXME: cast DATE_TYPE
        "DOD_HOSP"    : TIME_TYPE,  # FIXME: cast DATE_TYPE
        "DOD_SSN"     : TIME_TYPE,  # FIXME: cast DATE_TYPE
        "EXPIRE_FLAG" : INT8_TYPE,  # cast to bool
    },
    "PRESCRIPTIONS": {
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
    },
    "PROCEDUREEVENTS_MV": {
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
        "SECONDARYORDERCATEGORYNAME" : STRING_TYPE,  # All Null
        "ORDERCATEGORYDESCRIPTION"   : DICT_TYPE,
        "ISOPENBAG"                  : INT8_TYPE,  # cast to bool
        "CONTINUEINNEXTDEPT"         : INT8_TYPE,  # cast to bool
        "CANCELREASON"               : ID_TYPE,
        "STATUSDESCRIPTION"          : DICT_TYPE,
        "COMMENTS_EDITEDBY"          : DICT_TYPE,
        "COMMENTS_CANCELEDBY"        : DICT_TYPE,
        "COMMENTS_DATE"              : TIME_TYPE,
    },
    "PROCEDURES_ICD": {
        "ROW_ID"     : ID_TYPE,
        "SUBJECT_ID" : ID_TYPE,
        "HADM_ID"    : ID_TYPE,
        "SEQ_NUM"    : ID_TYPE,
        "ICD9_CODE"  : DICT_TYPE,
    },
    "SERVICES": {
        "ROW_ID"       : ID_TYPE,
        "SUBJECT_ID"   : ID_TYPE,
        "HADM_ID"      : ID_TYPE,
        "TRANSFERTIME" : TIME_TYPE,
        "PREV_SERVICE" : DICT_TYPE,
        "CURR_SERVICE" : DICT_TYPE,
    },
    "TRANSFERS": {
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
    },
}  # fmt: skip


class MIMIC_III(DatasetBase[MIMIC_III_Key, pl.LazyFrame]):
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
        `TIME_STAMP = round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36))`
        and `bin_k = 10`
        i.e. `TIME_STAMP = round(dt.total_seconds()*10/3600) = round(dt.total_hours()*10)`
        i.e. `TIME_STAMP ≈ 10*total_hours`
        so e.g. the last patient was roughly 250 hours, 10½ days.
    """

    DEFAULT_VERSION = "1.4"
    __version__: str  # pyright: ignore[reportIncompatibleMethodOverride]

    SOURCE_URL = r"https://physionet.org/content/mimiciii/get-zip/"
    INFO_URL = r"https://physionet.org/content/mimiciii/"
    HOME_URL = r"https://mimic.mit.edu/"

    table_names = list(get_args(MIMIC_III_Key.__value__))
    rawdata_hashes = {
        "mimic-iii-clinical-database-1.4.zip": "sha256:f9917f0f77f29d9abeb4149c96724618923a4725310c62fb75529a2c3e483abd"
    }

    def __post_init__(self) -> None:
        assert self.__version__ == "1.4"
        super().__post_init__()

    @property
    def rawdata_files(self) -> list[str]:  # type: ignore
        return [f"mimic-iii-clinical-database-{self.__version__}.zip"]

    @cached_property
    def filelist(self) -> dict[MIMIC_III_Key, str]:
        r"""Mapping between table_names and contents of the zip file."""
        if not self.version_info >= (1, 4):
            raise ValueError("MIMIC-III v1.4+ is required.")

        return {
            key: f"mimic-iii-clinical-database-{self.__version__}/{key}.csv.gz"
            for key in self.table_names
        }

    def clean_table(self, key: MIMIC_III_Key) -> None:
        with (
            ZipFile(self.rawdata_paths[self.rawdata_files[0]], "r") as archive,
            archive.open(self.filelist[key], "r") as compressed_file,
        ):
            schema = SCHEMAS[key]
            validate_schema(compressed_file, schema)
            table = pl.scan_csv(
                compressed_file,
                schema=schema,
            )

            try:
                table.sink_parquet(self.dataset_paths[key], engine="streaming")
            except BaseException:
                # Delete partial files if streaming fails.
                with suppress(FileNotFoundError):
                    self.dataset_paths[key].unlink()
                raise

    def load_table(self, key: MIMIC_III_Key, /) -> pl.LazyFrame:
        table = pl.scan_parquet(self.dataset_paths[key])
        table.collect_schema()
        return table

    def get_rawdata_file(self, fname: str, /) -> None:
        r"""Download a file from the MIMIC-III website."""
        if tuple(map(int, self.__version__.split("."))) < (1, 4):  # noqa: RUF048
            raise ValueError(
                "MIMIC-III v1.4+ is required. At the time of writing, the website"
                " does not provide legacy versions of the MIMIC-III dataset."
            )

        remote.download(
            self.SOURCE_URL + f"{self.__version__}/",
            self.rawdata_paths[fname],
            username=input("\nMIMIC-III username: "),
            password=getpass(prompt="MIMIC-III password: ", stream=None),
            headers={
                "User-Agent": "Wget/1.21.2"
            },  # NOTE: MIMIC only allows wget for some reason...
        )
