#!/usr/bin/env python
# coding: utf-8

# # MIMIC 4 data - dataset construction admissions

# In[1]:


import gzip
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pyarrow
import pyarrow.csv
import pyarrow.parquet

# # Load `admissions` table

# ## Table Schema

# In[2]:


rawdata_file = Path.cwd() / "mimic-iv-1.0.zip"
dataset_path = Path.cwd() / "processed"
rawdata_path = Path.cwd() / "raw"

files = {
    "admissions": "mimic-iv-1.0/core/admissions.csv.gz",
    "patients": "mimic-iv-1.0/core/patients.csv.gz",
    "chartevents": "mimic-iv-1.0/icu/chartevents.csv.gz",
}

CATEGORY = pyarrow.dictionary("int32", "string")
ID_TYPE = "int32"  # pyarrow.dictionary("int32", "int32", ordered=True)

column_types = {
    "admissions": {
        "subject_id": ID_TYPE,
        "hadm_id": ID_TYPE,
        "admittime": "timestamp[s]",
        "dischtime": "timestamp[s]",
        "deathtime": "timestamp[s]",
        "admission_type": CATEGORY,
        "admission_location": CATEGORY,
        "discharge_location": CATEGORY,
        "insurance": CATEGORY,
        "language": CATEGORY,
        "marital_status": CATEGORY,
        "ethnicity": CATEGORY,
        "edregtime": "timestamp[s]",
        "edouttime": "timestamp[s]",
        "hospital_expire_flag": "bool",
    },
    "patients": {
        "subject_id": ID_TYPE,
        "gender": CATEGORY,
        "anchor_age": "int32",
        "anchor_year": "int32",
        "anchor_year_group": CATEGORY,
        "dod": "timestamp[s]",
    },
    "chartevents": {
        "subject_id": ID_TYPE,
        "hadm_id": ID_TYPE,
        "stay_id": ID_TYPE,
        "itemid": ID_TYPE,
        "charttime": "timestamp[s]",
        "storetime": "timestamp[s]",
        # "value": CATEGORY,
        "valuenum": "float32",
        "valueuom": CATEGORY,
        "warning": "bool",
    },
}


null_values = [
    "-",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "?",
    "",
    "#N/A N/A",
    "#N/A",
    "#NA",
    "#na",
    "<N/A>",
    "<n/a>",
    "<NA>",
    "<na>",
    "1.#IND",
    "1.#QNAN",
    "INFORMATION NOT AVAILABLE",
    "N/A",
    "n/a",
    "NA",
    "na",
    "NAN",
    "NaN",
    "nan",
    "NONE",
    "None",
    "none",
    "NULL",
    "NULL",
    "Null",
    "null",
    "UNABLE TO OBTAIN",
    "UNKNOWN",
    "unknown",
]

types_map = {
    "string": pd.StringDtype(),
    "bool": pd.BooleanDtype(),
    "int8": pd.Int8Dtype(),
    "int16": pd.Int16Dtype(),
    "int32": pd.Int32Dtype(),
    "int64": pd.Int64Dtype(),
    "uint8": pd.UInt8Dtype(),
    "uint16": pd.UInt16Dtype(),
    "uint32": pd.UInt32Dtype(),
    "uint64": pd.UInt64Dtype(),
}


# ## Load `admissions` table

# In[3]:


key = "admissions"
with (
    ZipFile(rawdata_file) as archive,
    archive.open(files[key]) as compressed_file,
    gzip.open(compressed_file) as file,
):
    admissions = pyarrow.csv.read_csv(
        file,
        convert_options=pyarrow.csv.ConvertOptions(
            column_types=column_types[key],
            strings_can_be_null=True,
            null_values=null_values,
        ),
    )

admissions.shape, admissions.schema


# In[4]:


pyarrow.parquet.write_table(admissions, rawdata_path / f"{key}.parquet")
admissions = admissions.to_pandas(self_destruct=True, types_mapper=types_map.get)
admissions


# ## Load `patients` table

# In[5]:


key = "patients"
with (
    ZipFile(rawdata_file) as archive,
    archive.open(files[key]) as compressed_file,
    gzip.open(compressed_file) as file,
):
    patients = pyarrow.csv.read_csv(
        file,
        convert_options=pyarrow.csv.ConvertOptions(
            column_types=column_types[key],
            strings_can_be_null=True,
            null_values=null_values,
        ),
    )
patients.shape, patients.schema


# In[6]:


pyarrow.parquet.write_table(patients, rawdata_path / f"{key}.parquet")
patients = patients.to_pandas(self_destruct=True, types_mapper=types_map.get)
patients


# ## Load `chartevents` table

# In[7]:


# shape: (330M, 10) ⇝ 3.3B values
key = "chartevents"
with (
    ZipFile(rawdata_file) as archive,
    archive.open(files[key]) as compressed_file,
    gzip.open(compressed_file) as file,
):
    chartevents = pyarrow.csv.read_csv(
        file,
        convert_options=pyarrow.csv.ConvertOptions(
            column_types=column_types[key],
            strings_can_be_null=True,
            null_values=null_values,
        ),
    )

chartevents.shape, chartevents.schema


# In[8]:


pyarrow.parquet.write_table(chartevents, rawdata_path / f"{key}.parquet")
chartevents = chartevents.to_pandas(self_destruct=True, types_mapper=types_map.get)
chartevents


# In[ ]:


chartevents = chartevents.loc[
    chartevents.value.notna()
    | chartevents.valuenum.notna()
    | chartevents.valueuom.notna()
]


# In[ ]:


raise Exception


# In[ ]:


all_missing = c.sum(
    c.equal(
        c.add(
            c.add(
                c.cast(c.is_null(chartevents["value"]), "int64"),
                c.cast(c.is_null(chartevents["valuenum"]), "int64"),
            ),
            c.cast(c.is_null(chartevents["valueuom"]), "int64"),
        ),
        3,
    )
)


# In[ ]:


pyarrow.compute.sum(null_values)


# In[ ]:


float_values.drop_null


# In[ ]:


pyarrow.compute.sum(pyarrow.compute.equal(float_values, other_values).drop_null())


# In[ ]:


# In[ ]:


chartevents


# In[ ]:


float_mask = pyarrow.compute.utf8_is_numeric(chartevents["value"])


# In[ ]:


other_values = pyarrow.compute.filter(chartevents["valuenum"], float_mask)


# In[ ]:


float_values = pyarrow.compute.cast(
    pyarrow.compute.filter(chartevents["value"], float_mask), "float32"
)


# In[ ]:


null_values = pyarrow.compute.is_null(chartevents["value"])


# In[ ]:


from pyarrow import compute as c

# In[ ]:


all_missing


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# # Filter Dataset

# # Merge patients and admissions

# In[ ]:


admissions = pd.merge(admissions, patients, on="subject_id")


# ## Only keep patients with single admission

# In[ ]:


single_admissions = admissions.groupby("subject_id")["hadm_id"].nunique() == 1
selected_patients = single_admissions[single_admissions].index
mask = admissions["subject_id"].isin(selected_patients)
admissions = admissions[mask].copy()
print(f"Removing {(~mask).sum()} patients with multiple admissions!")
print(f"Number of patients   remaining: {admissions['subject_id'].nunique()}")
print(f"Number of admissions remaining: {admissions['hadm_id'].nunique()}")


# ## Only keep admissions with single patient associated

# In[ ]:


single_admissions = admissions.groupby("hadm_id")["subject_id"].nunique() == 1
selected_admissions = single_admissions[single_admissions].index
mask = admissions["hadm_id"].isin(selected_admissions)
admissions = admissions[mask].copy()
print(f"Removing {(~mask).sum()} admissions with multiple patients!")
print(f"Number of patients   remaining: {admissions['subject_id'].nunique()}")
print(f"Number of admissions remaining: {admissions['hadm_id'].nunique()}")


# ## Only keep patients that stayed between 2 and 29 days

# In[ ]:


admissions["elapsed_time"] = admissions["dischtime"] - admissions["admittime"]
elapsed_days = admissions["elapsed_time"].dt.days
admissions = admissions[(elapsed_days > 2) & (elapsed_days < 30)].copy()
print(f"Number of patients remainin in the dataframe: {admissions.shape}")


# ## Only keep patients older than 15

# In[ ]:


admissions = admissions[admissions["anchor_age"] > 15]
print(f"Number of patients remainin in the dataframe: {admissions.shape}")


# ## Only keep Patients that have time series data associated with them

# In[ ]:


admissions = admissions[admissions.hadm_id.isin(chartevents.hadm_id)]
print(f"Number of patients remainin in the dataframe: {admissions.shape}")


# # Serialize Pre-processed DataFrame

# In[ ]:


# Clean categories
def clean_categories(df):
    for col in df:
        if df[col].dtype == "category":
            df[col] = df[col].cat.remove_unused_categories()
    return df


admissions = clean_categories(admissions)
admissions.to_parquet(dataset_path / "admissions_processed.parquet")
admissions.shape, admissions.dtypes
