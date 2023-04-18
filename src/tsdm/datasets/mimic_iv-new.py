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

poe_detail:
    1. convert from tall to wide by unstacking field_name/field_value

prescriptions:
    1. drop rows whose dose_val_rx/form_val_disp is not float.

procedureevents:
    1. convert storetime to second resolution

chartevents:
    1. Drop rows with missing valueuom
    2. cast value to float
    3. unstack value/valueuom
"""

__all__ = ["MIMIC_IV"]

import os
import subprocess
from getpass import getpass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from tsdm.datasets.base import MultiFrameDataset

ID_TYPE = "uint32"
VALUE_TYPE = "float32"
TIME_TYPE = "timestamp[s]"
DATE_TYPE = "date32[day]"
BOOL_TYPE = "bool"
STRING_TYPE = "string"
DICT_TYPE = pa.dictionary("int32", "string")


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
TRUE_VALUES = ["Y", "Yes", "1"]
FALSE_VALUES = ["N", "No", "0"]

SCHEMA = {
    "icu": {
        "chartevents": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "stay_id": ID_TYPE,
            "caregiver_id": ID_TYPE,
            "itemid": ID_TYPE,
            "charttime": TIME_TYPE,
            "storetime": TIME_TYPE,
            "value": STRING_TYPE,  # string → float
            "valuenum": VALUE_TYPE,
            "valueuom": DICT_TYPE,
            "warning": BOOL_TYPE,
        },
        "inputevents": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "stay_id": ID_TYPE,
            "caregiver_id": ID_TYPE,
            "starttime": TIME_TYPE,
            "endtime": TIME_TYPE,
            "storetime": TIME_TYPE,
            "itemid": ID_TYPE,
            "amount": VALUE_TYPE,
            "amountuom": DICT_TYPE,
            "rate": VALUE_TYPE,
            "rateuom": DICT_TYPE,
            "orderid": ID_TYPE,
            "linkorderid": ID_TYPE,
            "ordercategoryname": DICT_TYPE,
            "secondaryordercategoryname": DICT_TYPE,
            "ordercomponenttypedescription": DICT_TYPE,
            "ordercategorydescription": DICT_TYPE,
            "patientweight": VALUE_TYPE,
            "totalamount": VALUE_TYPE,
            "totalamountuom": DICT_TYPE,
            "isopenbag": BOOL_TYPE,
            "continueinnextdept": BOOL_TYPE,
            "statusdescription": DICT_TYPE,
            "originalamount": VALUE_TYPE,
            "originalrate": VALUE_TYPE,
        },
        "outputevents": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "stay_id": ID_TYPE,
            "caregiver_id": ID_TYPE,
            "charttime": TIME_TYPE,
            "storetime": TIME_TYPE,
            "itemid": ID_TYPE,
            "value": VALUE_TYPE,
            "valueuom": DICT_TYPE,
        },
        "procedureevents": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "stay_id": ID_TYPE,
            "caregiver_id": ID_TYPE,
            "starttime": TIME_TYPE,
            "endtime": TIME_TYPE,
            # "storetime": TIME_TYPE,
            "itemid": ID_TYPE,
            "value": VALUE_TYPE,
            "valueuom": DICT_TYPE,
            "location": DICT_TYPE,
            "locationcategory": DICT_TYPE,
            "orderid": ID_TYPE,
            "linkorderid": ID_TYPE,
            "ordercategoryname": DICT_TYPE,
            "ordercategorydescription": DICT_TYPE,
            "patientweight": VALUE_TYPE,
            "isopenbag": BOOL_TYPE,
            "continueinnextdept": BOOL_TYPE,
            "statusdescription": DICT_TYPE,
            "originalamount": VALUE_TYPE,
            "originalrate": BOOL_TYPE,
        },
        "datetimeevents": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "stay_id": ID_TYPE,
            "caregiver_id": ID_TYPE,
            "charttime": TIME_TYPE,
            "storetime": TIME_TYPE,
            "itemid": ID_TYPE,
            "value": TIME_TYPE,
            "valueuom": DICT_TYPE,
            "warning": BOOL_TYPE,
        },
        "ingredientevents": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "stay_id": ID_TYPE,
            "caregiver_id": ID_TYPE,
            "starttime": TIME_TYPE,
            "endtime": TIME_TYPE,
            "storetime": TIME_TYPE,
            "itemid": ID_TYPE,
            "amount": VALUE_TYPE,
            "amountuom": DICT_TYPE,
            "rate": VALUE_TYPE,
            "rateuom": DICT_TYPE,
            "orderid": ID_TYPE,
            "linkorderid": ID_TYPE,
            "statusdescription": DICT_TYPE,
            "originalamount": VALUE_TYPE,
            "originalrate": VALUE_TYPE,
        },
        "icustays": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "stay_id": ID_TYPE,
            "first_careunit": DICT_TYPE,
            "last_careunit": DICT_TYPE,
            "intime": TIME_TYPE,
            "outtime": TIME_TYPE,
            "los": VALUE_TYPE,
        },
        "d_items": {
            "itemid": ID_TYPE,
            "label": STRING_TYPE,
            "abbreviation": STRING_TYPE,
            "linksto": DICT_TYPE,
            "category": DICT_TYPE,
            "unitname": DICT_TYPE,
            "param_type": DICT_TYPE,
            "lownormalvalue": VALUE_TYPE,
            "highnormalvalue": VALUE_TYPE,
        },
        "caregiver": {
            "caregiver_id": ID_TYPE,
        },
    },
    "hosp": {
        "transfers": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "transfer_id": ID_TYPE,
            "eventtype": DICT_TYPE,
            "careunit": DICT_TYPE,
            "intime": TIME_TYPE,
            "outtime": TIME_TYPE,
        },
        "admissions": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "admittime": TIME_TYPE,
            "dischtime": TIME_TYPE,
            "deathtime": TIME_TYPE,
            "admission_type": DICT_TYPE,
            "admit_provider_id": DICT_TYPE,
            "admission_location": DICT_TYPE,
            "discharge_location": DICT_TYPE,
            "insurance": DICT_TYPE,
            "language": DICT_TYPE,
            "marital_status": DICT_TYPE,
            "race": DICT_TYPE,
            "edregtime": TIME_TYPE,
            "edouttime": TIME_TYPE,
            "hospital_expire_flag": BOOL_TYPE,
        },
        "services": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "transfertime": TIME_TYPE,
            "prev_service": DICT_TYPE,
            "curr_service": DICT_TYPE,
        },
        "provider": {
            "provider_id": STRING_TYPE,
        },
        "procedures_icd": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "seq_num": ID_TYPE,
            "chartdate": DATE_TYPE,
            "icd_code": DICT_TYPE,
            "icd_version": ID_TYPE,
        },
        "prescriptions": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "pharmacy_id": ID_TYPE,
            "poe_id": STRING_TYPE,
            "poe_seq": ID_TYPE,
            "order_provider_id": DICT_TYPE,
            "starttime": TIME_TYPE,
            "stoptime": TIME_TYPE,
            "drug_type": DICT_TYPE,
            "drug": DICT_TYPE,
            "formulary_drug_cd": DICT_TYPE,
            "gsn": DICT_TYPE,
            "ndc": DICT_TYPE,
            "prod_strength": DICT_TYPE,
            "form_rx": DICT_TYPE,
            "dose_val_rx": STRING_TYPE,  # string → float
            "dose_unit_rx": DICT_TYPE,
            "form_val_disp": STRING_TYPE,  # string → float
            "form_unit_disp": DICT_TYPE,
            "doses_per_24_hrs": VALUE_TYPE,
            "route": DICT_TYPE,
        },
        "poe_detail": {
            "poe_id": STRING_TYPE,
            "poe_seq": ID_TYPE,
            "subject_id": ID_TYPE,
            "field_name": DICT_TYPE,
            "field_value": STRING_TYPE,  # unstack column
        },
        "poe": {
            "poe_id": STRING_TYPE,
            "poe_seq": ID_TYPE,
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "ordertime": TIME_TYPE,
            "order_type": DICT_TYPE,
            "order_subtype": DICT_TYPE,
            "transaction_type": DICT_TYPE,
            "discontinue_of_poe_id": STRING_TYPE,
            "discontinued_by_poe_id": STRING_TYPE,
            "order_provider_id": DICT_TYPE,
            "order_status": DICT_TYPE,
        },
        "pharmacy": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "pharmacy_id": ID_TYPE,
            "poe_id": STRING_TYPE,
            "starttime": TIME_TYPE,
            "stoptime": TIME_TYPE,
            "medication": DICT_TYPE,
            "proc_type": DICT_TYPE,
            "status": DICT_TYPE,
            "entertime": TIME_TYPE,
            "verifiedtime": TIME_TYPE,
            "route": DICT_TYPE,
            "frequency": DICT_TYPE,
            "disp_sched": DICT_TYPE,
            "infusion_type": DICT_TYPE,
            "sliding_scale": BOOL_TYPE,
            "lockout_interval": DICT_TYPE,
            "basal_rate": VALUE_TYPE,
            "one_hr_max": DICT_TYPE,
            "doses_per_24_hrs": VALUE_TYPE,
            "duration": VALUE_TYPE,
            "duration_interval": DICT_TYPE,
            "expiration_value": VALUE_TYPE,
            "expiration_unit": DICT_TYPE,
            "expirationdate": TIME_TYPE,
            "dispensation": DICT_TYPE,
            "fill_quantity": DICT_TYPE,
        },
        "patients": {
            "subject_id": ID_TYPE,
            "gender": DICT_TYPE,
            "anchor_age": ID_TYPE,
            "anchor_year": ID_TYPE,
            "anchor_year_group": DICT_TYPE,
            "dod": DATE_TYPE,
        },
        "omr": {
            "subject_id": ID_TYPE,
            "chartdate": DATE_TYPE,
            "seq_num": ID_TYPE,
            "result_name": STRING_TYPE,  # unstack
            "result_value": STRING_TYPE,  # unstack
            # split blood pressure into 2 floats (blood pressure systolic/diastolic).
        },
        "microbiologyevents": {
            "microevent_id": ID_TYPE,
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "micro_specimen_id": ID_TYPE,
            "order_provider_id": DICT_TYPE,
            "chartdate": TIME_TYPE,
            "charttime": TIME_TYPE,
            "spec_itemid": ID_TYPE,
            "spec_type_desc": DICT_TYPE,
            "test_seq": ID_TYPE,
            "storedate": TIME_TYPE,
            "storetime": TIME_TYPE,
            "test_itemid": ID_TYPE,
            "test_name": DICT_TYPE,
            "org_itemid": ID_TYPE,
            "org_name": DICT_TYPE,
            "isolate_num": ID_TYPE,
            "quantity": DICT_TYPE,
            "ab_itemid": ID_TYPE,
            "ab_name": DICT_TYPE,
            "dilution_text": STRING_TYPE,  # convert to float
            "dilution_comparison": DICT_TYPE,
            "dilution_value": VALUE_TYPE,
            "interpretation": DICT_TYPE,
            "comments": STRING_TYPE,
        },
        "labevents": {
            "labevent_id": ID_TYPE,
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "specimen_id": ID_TYPE,
            "itemid": ID_TYPE,
            "order_provider_id": DICT_TYPE,
            "charttime": TIME_TYPE,
            "storetime": TIME_TYPE,
            "value": STRING_TYPE,  # cast Float32
            "valuenum": VALUE_TYPE,
            "valueuom": DICT_TYPE,
            "ref_range_lower": VALUE_TYPE,
            "ref_range_upper": VALUE_TYPE,
            "flag": DICT_TYPE,
            "priority": DICT_TYPE,
            "comments": STRING_TYPE,
        },
        "hcpcsevents": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "chartdate": DATE_TYPE,
            "hcpcs_cd": DICT_TYPE,
            "seq_num": ID_TYPE,
            "short_description": DICT_TYPE,
        },
        "emar_detail": {
            "subject_id": ID_TYPE,
            "emar_id": STRING_TYPE,
            "emar_seq": ID_TYPE,
            "parent_field_ordinal": DICT_TYPE,
            "administration_type": DICT_TYPE,
            "pharmacy_id": ID_TYPE,
            "barcode_type": DICT_TYPE,
            "reason_for_no_barcode": DICT_TYPE,
            "complete_dose_not_given": BOOL_TYPE,
            "dose_due": STRING_TYPE,  # cast float
            "dose_due_unit": DICT_TYPE,
            "dose_given": STRING_TYPE,  # cast float
            "dose_given_unit": DICT_TYPE,
            "will_remainder_of_dose_be_given": BOOL_TYPE,
            "product_amount_given": STRING_TYPE,  # cast float
            "product_unit": DICT_TYPE,
            "product_code": DICT_TYPE,
            "product_description": DICT_TYPE,
            "product_description_other": DICT_TYPE,
            "prior_infusion_rate": STRING_TYPE,  # cast float
            "infusion_rate": STRING_TYPE,  # cast float
            "infusion_rate_adjustment": DICT_TYPE,
            "infusion_rate_adjustment_amount": STRING_TYPE,  # cast float
            "infusion_rate_unit": DICT_TYPE,
            "route": DICT_TYPE,
            "infusion_complete": BOOL_TYPE,
            "completion_interval": DICT_TYPE,
            "new_iv_bag_hung": BOOL_TYPE,
            "continued_infusion_in_other_location": BOOL_TYPE,
            "restart_interval": DICT_TYPE,
            "side": DICT_TYPE,
            "site": DICT_TYPE,
            "non_formulary_visual_verification": BOOL_TYPE,
        },
        "emar": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "emar_id": STRING_TYPE,
            "emar_seq": ID_TYPE,
            "poe_id": STRING_TYPE,
            "pharmacy_id": ID_TYPE,
            "enter_provider_id": DICT_TYPE,
            "charttime": TIME_TYPE,
            "medication": DICT_TYPE,
            "event_txt": DICT_TYPE,
            "scheduletime": TIME_TYPE,
            "storetime": TIME_TYPE,
        },
        "drgcodes": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "drg_type": DICT_TYPE,
            "drg_code": ID_TYPE,
            "description": DICT_TYPE,
            "drg_severity": ID_TYPE,
            "drg_mortality": ID_TYPE,
        },
        "diagnoses_icd": {
            "subject_id": ID_TYPE,
            "hadm_id": ID_TYPE,
            "seq_num": ID_TYPE,
            "icd_code": DICT_TYPE,
            "icd_version": ID_TYPE,
        },
        "d_labitems": {
            "itemid": ID_TYPE,
            "label": STRING_TYPE,
            "fluid": DICT_TYPE,
            "category": DICT_TYPE,
        },
        "d_icd_procedures": {
            "icd_code": STRING_TYPE,
            "icd_version": ID_TYPE,
            "long_title": STRING_TYPE,
        },
        "d_icd_diagnoses": {
            "icd_code": STRING_TYPE,
            "icd_version": ID_TYPE,
            "long_title": STRING_TYPE,
        },
        "d_hcpcs": {
            "code": STRING_TYPE,
            "category": ID_TYPE,
            "long_description": STRING_TYPE,
            "short_description": DICT_TYPE,
        },
    },
}


class MIMIC_IV(MultiFrameDataset):
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

    BASE_URL = r"https://www.physionet.org/content/mimiciv/get-zip/1.0/"
    INFO_URL = r"https://www.physionet.org/content/mimiciv/1.0/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/mbilos/neural-flows-experiments"
    VERSION = r"1.0"
    RAWDATA_HASH = "dd226e8694ad75149eed2840a813c24d5c82cac2218822bc35ef72e900baad3d"

    rawdata_files = "mimic-iv-1.0.zip"
    rawdata_paths: Path

    # fmt: off
    internal_files = {
        "admissions"         : "mimic-iv-1.0/core/admissions.csv.gz",
        "patients"           : "mimic-iv-1.0/core/patients.csv.gz",
        "transfers"          : "mimic-iv-1.0/core/transfers.csv.gz",
        "chartevents"        : "mimic-iv-1.0/icu/chartevents.csv.gz",
        "datetimeevents"     : "mimic-iv-1.0/icu/datetimeevents.csv.gz",
        "d_items"            : "mimic-iv-1.0/icu/d_items.csv.gz",
        "icustays"           : "mimic-iv-1.0/icu/icustays.csv.gz",
        "inputevents"        : "mimic-iv-1.0/icu/inputevents.csv.gz",
        "outputevents"       : "mimic-iv-1.0/icu/outputevents.csv.gz",
        "procedureevents"    : "mimic-iv-1.0/icu/procedureevents.csv.gz",
        "d_hcpcs"            : "mimic-iv-1.0/hosp/d_hcpcs.csv.gz",
        "diagnoses_icd"      : "mimic-iv-1.0/hosp/diagnoses_icd.csv.gz",
        "d_icd_diagnoses"    : "mimic-iv-1.0/hosp/d_icd_diagnoses.csv.gz",
        "d_icd_procedures"   : "mimic-iv-1.0/hosp/d_icd_procedures.csv.gz",
        "d_labitems"         : "mimic-iv-1.0/hosp/d_labitems.csv.gz",
        "drgcodes"           : "mimic-iv-1.0/hosp/drgcodes.csv.gz",
        "emar"               : "mimic-iv-1.0/hosp/emar.csv.gz",
        "emar_detail"        : "mimic-iv-1.0/hosp/emar_detail.csv.gz",
        "hcpcsevents"        : "mimic-iv-1.0/hosp/hcpcsevents.csv.gz",
        "labevents"          : "mimic-iv-1.0/hosp/labevents.csv.gz",
        "microbiologyevents" : "mimic-iv-1.0/hosp/microbiologyevents.csv.gz",
        "pharmacy"           : "mimic-iv-1.0/hosp/pharmacy.csv.gz",
        "poe"                : "mimic-iv-1.0/hosp/poe.csv.gz",
        "poe_detail"         : "mimic-iv-1.0/hosp/poe_detail.csv.gz",
        "prescriptions"      : "mimic-iv-1.0/hosp/prescriptions.csv.gz",
        "procedures_icd"     : "mimic-iv-1.0/hosp/procedures_icd.csv.gz",
        "services"           : "mimic-iv-1.0/hosp/services.csv.gz",
    }
    # fmt: on

    KEYS = list(internal_files.keys())

    def clean_table(self, key: str) -> None:
        ...

    def load_table(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.dataset_paths[key])

    def download_table(self, **_: Any) -> None:
        cut_dirs = self.BASE_URL.count("/") - 3
        user = input("MIMIC-IV username: ")
        password = getpass(prompt="MIMIC-IV password: ", stream=None)

        os.environ["PASSWORD"] = password

        subprocess.run(
            f"wget --user {user} --password $PASSWORD -c -r -np -nH -N "
            + f"--cut-dirs {cut_dirs} -P {self.RAWDATA_DIR!r} {self.BASE_URL} -O {self.rawdata_paths}",
            shell=True,
            check=True,
        )

        file = self.RAWDATA_DIR / "index.html"
        os.rename(file, self.rawdata_files)
