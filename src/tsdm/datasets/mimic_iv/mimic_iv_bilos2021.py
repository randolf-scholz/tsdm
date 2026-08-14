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
"""

__all__ = [
    "RAWDATA_SCHEMA",
    "TARGET_SCHEMA",
    "MIMIC_IV_Bilos2021_FromPreprocessed",
    "LAB_LABELS",
    "PRESCRIPTION_LABELS",
    "LABEL_NAMES",
    "LABELS",
    "OUTPUT_LABELS",
    "MIMIC_IV_Bilos2021",
]


from typing import Literal

import polars as pl

from tsdm.datasets.base import DatasetBase
from tsdm.datatools import validate_schema

from .mimic_iv import MIMIC_IV

# original order from their pre-processing code, which is not sorted by label number.
LABELS = [  # 0...101
     3,   4,   5,   6,   7,   8,   9,  20,  22,   0,  25,  26,  27,  28,  10,  11,  12,
    13,  14,  21,  15,  17,  16,  18,  19,  35,  36,  23,  24,  46,  47,  48,  49,  31,
    52,  53,  54,  32,  34,  55,   2,  56,  51,  37,  57,  58,  59,  60,  61,  62,  63,
    64,  65,  39,  38,  40,  41,  66,  67,  44,  68,  69,  70,  71,  72,  29,  30,  73,
    74,  75,  76,  77,  78,  79,  43,  80,  33,  81,  42,  82,  83,  84,  85,  86,  87,
    88,  89,   1,  94,  91,  95,  90,  96,  98,  92,  45,  97,  50,  93, 101,  99, 100,
]  # fmt: skip
RAWDATA_SCHEMA = {
    "hadm_id": pl.UInt32,
    "time_stamp": pl.Int16,
    **{
        f"{kind}_label_{label}": pl.Float32 if kind == "Value" else pl.UInt8
        for label in LABELS
        for kind in ("Value", "Mask")
    },
}
TARGET_SCHEMA = {
    "hadm_id": pl.UInt32,
    "time_stamp": pl.Int16,
    # The 5σ outlier filter removes every observation from labels 37 and 71.
    **{f"Value_{label}": pl.Float32 for label in range(102) if label not in (37, 71)},
}

type Key = Literal["raw_timeseries", "timeseries"]


class MIMIC_IV_Bilos2021_FromPreprocessed(DatasetBase[Key, pl.DataFrame]):
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

    Attributes:
        raw_timeseries: The original Bilos et al. ``full_dataset.csv`` export,
            converted to parquet with its value and mask columns intact.
        timeseries: ``raw_timeseries`` with masks folded into the value columns,
            empty rows dropped, every variable standardized over the full dataset,
            and 5σ outliers removed.

    References:
        - | Neural Flows: Efficient Alternative to Neural ODEs
          | Biloš et al.
          | NeurIPS 2021
          | https://proceedings.neurips.cc/paper/2021/hash/b21f9f98829dea9a48fd8aaddc1f159d-Abstract.html
    """

    SOURCE_URL = r"https://physionet.org/content/mimiciv/get-zip/1.0/"
    INFO_URL = r"https://physionet.org/content/mimiciv/1.0/"
    HOME_URL = r"https://mimic.mit.edu/"
    GITHUB_URL = r"https://github.com/mbilos/neural-flows-experiments"

    table_names = ["raw_timeseries", "timeseries"]  # pyright: ignore[reportAssignmentType]
    rawdata_files = ["full_dataset.csv"]
    rawdata_hashes = {
        "full_dataset.csv": "sha256:bf0f7cecd6b0eb81fb2cce2094c8454fbe19ac1b103d86705610fe972a14c1f0"
    }
    rawdata_schemas = {"full_dataset.csv": RAWDATA_SCHEMA}
    rawdata_shapes = {"full_dataset.csv": (2_485_649, 206)}
    table_schemas = {
        "raw_timeseries": RAWDATA_SCHEMA,
        "timeseries": TARGET_SCHEMA,
    }
    table_shapes = {
        "raw_timeseries": (2_485_649, 206),
        "timeseries": (2_485_649, 102),
    }

    def clean_raw_timeseries(self) -> pl.DataFrame:
        r"""Convert the Bilos et al. ``full_dataset.csv`` export to parquet."""
        fname = "full_dataset.csv"
        rawdata_schema = self.rawdata_schemas[fname]
        rawdata_shape = self.rawdata_shapes[fname]
        rawdata_path = self.rawdata_paths[fname]
        validate_schema(rawdata_path, rawdata_schema)
        csv_schema = {**rawdata_schema, "hadm_id": pl.Float64}
        table = (
            pl.read_csv(rawdata_path, schema=csv_schema)
            .with_columns(pl.col("hadm_id").cast(pl.UInt32))
            .fill_nan(None)
        )

        if table.shape != rawdata_shape:
            raise ValueError(f"{table.shape=} does not match {rawdata_shape=}.")

        return table

    def clean_timeseries(self) -> pl.DataFrame:
        r"""Apply the masking, normalization, and 5σ filtering from Bilos et al."""
        if not self.dataset_files_exist("raw_timeseries"):
            self.clean("raw_timeseries")
        table = pl.scan_parquet(self.dataset_paths["raw_timeseries"]).fill_nan(None)

        value_columns = [f"Value_label_{label}" for label in range(102)]
        mask_columns = [f"Mask_label_{label}" for label in range(102)]
        target_columns = [f"Value_{label}" for label in range(102)]
        columns = table.collect_schema().names()

        if missing_values := set(value_columns) - set(columns):
            raise ValueError(f"Value columns not found: {missing_values}")

        if missing_masks := set(mask_columns) - set(columns):
            raise ValueError(f"Mask columns not found: {missing_masks}")

        # fold masks into value column and rename.
        masked = table.select(
            pl.col("hadm_id").cast(pl.Int32),
            pl.col("time_stamp"),
            *(
                pl.when(pl.col(f"Mask_label_{k}").eq(1))
                .then(pl.col(f"Value_label_{k}"))
                .otherwise(None)
                .alias(f"Value_{k}")
                for k in range(102)
            ),
        ).filter(
            pl.any_horizontal(pl.col(column).is_not_null() for column in target_columns)
        )

        # NOTE: For the MIMIC-III and MIMIC-IV datasets, Bilos et al. perform standardization
        #  over the full data slice, including test!
        # https://github.com/mbilos/neural-flows-experiments/blob/master/nfe/experiments/gru_ode_bayes/lib/get_data.py
        normalized = masked.select(
            "hadm_id",
            "time_stamp",
            *(
                ((pl.col(column) - pl.col(column).mean()) / pl.col(column).std(ddof=1))
                for column in target_columns
            ),
        )

        # NOTE: For the MIMIC-IV dataset, Bilos et al. drop 5σ-outliers.
        table = normalized.select(
            "hadm_id",
            "time_stamp",
            *(
                pl.when(pl.col(column).is_between(-5, 5, closed="none"))
                .then(pl.col(column))
                .otherwise(None)
                for column in target_columns
            ),
        ).collect()

        return table.select(*TARGET_SCHEMA).sort("hadm_id", "time_stamp")

    def load_table(
        self, key: Literal["raw_timeseries", "timeseries"], /
    ) -> pl.DataFrame:
        r"""Load a cleaned table as a Polars DataFrame."""
        return pl.read_parquet(self.dataset_paths[key])

    def get_rawdata_file(self, _: str, /) -> None:
        fname = self.rawdata_files[0]
        if not self.rawdata_files_exist():
            raise RuntimeError(
                "Please manually apply the preprocessing code found at"
                f" {self.GITHUB_URL}.\nPut the resulting file {fname!r} in {self.RAWDATA_DIR}."
            )


# Labels indexed by the feature code that was assigned in Bilos et al.'s
# ``datamerging.ipynb``. Keeping this mapping explicit makes the output stable
# despite the unordered nature of lazy query execution.
LABEL_NAMES = [
    "PO Intake",
    "Heparin Sodium",
    "Dextrose 5%",
    "Hematocrit",
    "Platelet Count",
    "Creatinine",
    "Magnesium",
    "Urea Nitrogen",
    "PTT",
    "White Blood Cells",
    "Hemoglobin",
    "MCH",
    "MCV",
    "RDW",
    "Red Blood Cells",
    "Bicarbonate",
    "Chloride",
    "Calcium, Total",
    "Phosphate",
    "PT",
    "Potassium",
    "Anion Gap",
    "Sodium",
    "Glucose",
    "Void",
    "Cefazolin",
    "Hydralazine",
    "Pantoprazole (Protonix)",
    "Solution",
    "Hydromorphone (Dilaudid)",
    "OR Crystalloid Intake",
    "Vancomycin",
    "Heparin Sodium (Prophylaxis)",
    "Insulin - Regular",
    "NaCl 0.9%",
    "pH",
    "Specific Gravity",
    "Base Excess",
    "Calculated Total CO2",
    "Lactate",
    "pCO2",
    "pO2",
    "OR EBL",
    "OR Urine",
    "Foley",
    "Emesis",
    "Potassium Chloride",
    "Gastric Meds",
    "GT Flush",
    "Furosemide (Lasix)",
    "Free Water",
    "Propofol",
    "Famotidine (Pepcid)",
    "Pre-Admission/Non-ICU Intake",
    "Piperacillin/Tazobactam (Zosyn)",
    "Norepinephrine",
    "LR",
    "Alanine Aminotransferase (ALT)",
    "Alkaline Phosphatase",
    "Asparate Aminotransferase (AST)",
    "Bilirubin, Total",
    "Basophils",
    "Eosinophils",
    "Lymphocytes",
    "Monocytes",
    "Neutrophils",
    "Albumin",
    "Pre-Admission",
    "Acetaminophen-IV",
    "Magnesium Sulfate",
    "Insulin - Glargine",
    "Insulin - Humalog",
    "Fentanyl",
    "Fentanyl (Concentrate)",
    "Magnesium Sulfate (Bolus)",
    "K Phos",
    "Calcium Gluconate",
    "Metronidazole",
    "Cefepime",
    "Piggyback",
    "Morphine Sulfate",
    "Sterile Water",
    "OR Cell Saver Intake",
    "KCL (Bolus)",
    "Albumin 5%",
    "Packed Red Blood Cells",
    "Dexmedetomidine (Precedex)",
    "Phenylephrine",
    "Nitroglycerin",
    "Oral Gastric",
    "Lorazepam (Ativan)",
    "Metoprolol",
    "D5 1/2NS",
    "Straight Cath",
    "TF Residual",
    "Ceftriaxone",
    "Midazolam (Versed)",
    "Nasogastric",
    "Pantoprazole (Protonix) Continuous",
    "Stool",
    "TF Residual Output",
    "Fecal Bag",
]

LAB_LABELS = {
    "Albumin",
    "Alanine Aminotransferase (ALT)",
    "Alkaline Phosphatase",
    "Anion Gap",
    "Asparate Aminotransferase (AST)",
    "Base Excess",
    "Basophils",
    "Bicarbonate",
    "Bilirubin, Total",
    "Calcium, Total",
    "Calculated Total CO2",
    "Chloride",
    "Creatinine",
    "Eosinophils",
    "Glucose",
    "Hematocrit",
    "Hemoglobin",
    "Lactate",
    "Lymphocytes",
    "MCH",
    "MCV",
    "Magnesium",
    "Monocytes",
    "Neutrophils",
    "PT",
    "PTT",
    "Phosphate",
    "Platelet Count",
    "Potassium",
    "RDW",
    "Red Blood Cells",
    "Sodium",
    "Specific Gravity",
    "Urea Nitrogen",
    "White Blood Cells",
    "pCO2",
    "pH",
    "pO2",
}

OUTPUT_LABELS = {
    "Foley",
    "Void",
    "OR Urine",
    "Chest Tube",
    "Oral Gastric",
    "Pre-Admission",
    "TF Residual",
    "OR EBL",
    "Emesis",
    "Nasogastric",
    "Stool",
    "Jackson Pratt",
    "TF Residual Output",
    "Fecal Bag",
    "Straight Cath",
}

PRESCRIPTION_LABELS = {
    "Acetaminophen",
    "Aspirin",
    "Bisacodyl",
    "Insulin",
    "Heparin",
    "Docusate Sodium",
    "D5W",
    "Humulin-R Insulin",
    "Potassium Chloride",
    "Magnesium Sulfate",
    "Metoprolol Tartrate",
    "Sodium Chloride 0.9%  Flush",
    "Pantoprazole",
}


class MIMIC_IV_Bilos2021(DatasetBase[Key, pl.DataFrame]):
    r"""Polars reimplementation of the MIMIC-IV preprocessing by Bilos et al.

    The raw MIMIC-IV tables are supplied by :class:`MIMIC_IV`. They must already
    have been converted from the source archive to parquet by that dataset. This
    dataset only materializes the final, normalized time-series table.

    Attributes:
        raw_timeseries: The Polars recreation of the Bilos et al.
            ``full_dataset.csv`` export, including its value and mask columns.
        timeseries: ``raw_timeseries`` with masks folded into the value columns,
            empty rows dropped, every variable standardized over the full dataset,
            and 5σ outliers removed.

    References:
        - | Neural Flows: Efficient Alternative to Neural ODEs
          | Biloš et al.
          | NeurIPS 2021
          | https://proceedings.neurips.cc/paper/2021/hash/b21f9f98829dea9a48fd8aaddc1f159d-Abstract.html
    """

    SOURCE_URL = r"https://physionet.org/content/mimiciv/get-zip/1.0/"
    INFO_URL = MIMIC_IV.INFO_URL
    HOME_URL = MIMIC_IV.HOME_URL
    GITHUB_URL = r"https://github.com/mbilos/neural-flows-experiments"

    rawdata_files = []
    table_names = ["raw_timeseries", "timeseries"]  # pyright: ignore[reportAssignmentType]
    table_schemas = {
        "raw_timeseries": RAWDATA_SCHEMA,
        "timeseries": TARGET_SCHEMA,
    }
    table_shapes = {
        "raw_timeseries": (2_485_649, 206),
        "timeseries": (2_485_649, 102),
    }

    def __post_init__(self) -> None:
        r"""Use the already-preprocessed MIMIC-IV source tables."""
        self.raw_dataset = MIMIC_IV(
            version="1.0", initialize=False, verbose=self.verbose
        )

    def preprocess_admissions(self) -> pl.DataFrame:
        r"""Select the admissions retained by ``admissions.ipynb``.

        Patients must have exactly one admission, be older than 15, stay from
        three up to (but excluding) 30 days, and have at least one chart event.
        """
        admissions = self.raw_dataset.load_table("admissions")
        single_admission_subjects = (
            admissions.group_by("subject_id")
            .agg(pl.col("hadm_id").n_unique().alias("n_admissions"))
            .filter(pl.col("n_admissions").eq(1))
            .select("subject_id")
        )
        patients = self.raw_dataset.load_table("patients").select(
            "subject_id", "anchor_age"
        )
        charted_admissions = (
            self.raw_dataset.load_table("chartevents").select("hadm_id").unique()
        )
        elapsed_time = pl.col("dischtime") - pl.col("admittime")

        return (
            admissions.join(single_admission_subjects, on="subject_id", how="semi")
            .join(patients, on="subject_id", how="inner")
            .with_columns(elapsed_time.alias("elapsed_time"))
            .filter(elapsed_time.is_between(pl.duration(days=3), pl.duration(days=30)))
            .filter(pl.col("anchor_age").gt(15))
            .join(charted_admissions, on="hadm_id", how="semi")
            .collect()
        )

    def preprocess_inputevents(self) -> pl.DataFrame:
        r"""Clean and discretize the selected medication input events."""
        admission_ids = self.preprocess_admissions().select("hadm_id").lazy()
        inputevents = (
            self.raw_dataset.load_table("inputevents")
            .join(admission_ids, on="hadm_id", how="semi")
            .select(
                "subject_id",
                "hadm_id",
                "starttime",
                "endtime",
                "itemid",
                "amount",
                "amountuom",
                "rate",
                "rateuom",
                "patientweight",
                "ordercategorydescription",
            )
            .join(
                self.raw_dataset.load_table("d_items").select("itemid", "label"),
                on="itemid",
                how="inner",
            )
            .with_columns(
                pl.col("label").cast(pl.String),
                pl.col("amountuom").cast(pl.String),
                pl.col("rateuom").cast(pl.String),
                pl.col("ordercategorydescription").cast(pl.String),
            )
        )
        common_labels = (
            inputevents.group_by("label")
            .agg(pl.col("subject_id").n_unique().alias("n_subjects"))
            .sort("n_subjects", descending=True, maintain_order=True)
            .head(50)
            .select("label")
        )
        inputevents = inputevents.join(common_labels, on="label", how="semi")

        pantoprazole_continuous = pl.col("itemid").eq(225910) & pl.col(
            "ordercategorydescription"
        ).eq("Continuous Med")
        inputevents = inputevents.with_columns(
            pl.when(pantoprazole_continuous)
            .then(pl.lit("Pantoprazole (Protonix) Continuous"))
            .otherwise(pl.col("label"))
            .alias("label"),
            pl.when(pantoprazole_continuous)
            .then(pl.lit(2217441))
            .otherwise(pl.col("itemid"))
            .alias("itemid"),
        )

        inputevents = _keep_matching_units(
            inputevents,
            column="amountuom",
            expected={
                "Cefepime": "dose",
                "Ceftriaxone": "dose",
                "Ciprofloxacin": "dose",
                "Famotidine (Pepcid)": "dose",
                "Fentanyl (Concentrate)": "mg",
                "Heparin Sodium (Prophylaxis)": "dose",
                "Hydromorphone (Dilaudid)": "mg",
                "Magnesium Sulfate": "grams",
                "Metoprolol": "mg",
                "Metronidazole": "dose",
                "Pantoprazole (Protonix)": "dose",
                "Piperacillin/Tazobactam (Zosyn)": "dose",
                "Propofol": "mg",
                "Ranitidine (Prophylaxis)": "dose",
                "Vancomycin": "dose",
                "Acetaminophen-IV": "mg",
                "D5 1/2NS": "ml",
                "LR": "ml",
                "NaCl 0.9%": "ml",
                "OR Crystalloid Intake": "ml",
                "PO Intake": "ml",
                "Pre-Admission/Non-ICU Intake": "ml",
            },
        ).filter(pl.col("itemid").ne(225850) | pl.col("amountuom").eq("dose"))
        fentanyl_concentrate = pl.col("label").eq("Fentanyl (Concentrate)") & pl.col(
            "amountuom"
        ).eq("mg")
        fentanyl = pl.col("itemid").eq(221744) & pl.col("amountuom").eq("mg")
        dexmedetomidine = pl.col("label").eq("Dexmedetomidine (Precedex)") & pl.col(
            "amountuom"
        ).eq("mcg")
        inputevents = inputevents.with_columns(
            pl.when(fentanyl_concentrate | fentanyl)
            .then(pl.col("amount") * 1000)
            .when(dexmedetomidine)
            .then(pl.col("amount") / 1000)
            .otherwise(pl.col("amount"))
            .alias("amount"),
            pl.when(fentanyl_concentrate | fentanyl)
            .then(pl.lit("mcg"))
            .when(dexmedetomidine)
            .then(pl.lit("mg"))
            .otherwise(pl.col("amountuom"))
            .alias("amountuom"),
        )
        inputevents = _keep_matching_units(
            inputevents,
            column="rateuom",
            expected={
                "Acetaminophen-IV": "mg/min",
                "Dextrose 5%": "mL/hour",
                "Fentanyl (Concentrate)": "mcg/hour",
                "Magnesium Sulfate (Bolus)": "mL/hour",
                "NaCl 0.9%": "mL/hour",
                "Packed Red Blood Cells": "mL/hour",
                "Phenylephrine": "mcg/kg/min",
                "Piggyback": "mL/hour",
                "Sterile Water": "mL/hour",
            },
        )

        duration = pl.col("endtime") - pl.col("starttime")
        is_long = duration.gt(pl.duration(minutes=30)).fill_null(pl.lit(value=False))
        repeats = (duration.dt.total_seconds() / (30 * 60)).ceil().cast(pl.UInt32)
        spread_inputs = (
            inputevents.filter(is_long)
            .with_columns(repeats.alias("_repeats"))
            .with_columns(pl.int_ranges(0, "_repeats").alias("_step"))
            .explode("_step")
            .with_columns(
                (pl.col("amount") / pl.col("_repeats")).alias("amount"),
                (pl.col("starttime") + pl.duration(minutes=30) * pl.col("_step")).alias(
                    "charttime"
                ),
            )
            .drop("_repeats", "_step")
        )
        point_inputs = inputevents.filter(~is_long).with_columns(
            pl.col("starttime").alias("charttime")
        )
        return (
            pl.concat((spread_inputs, point_inputs), how="vertical_relaxed")
            .select("subject_id", "hadm_id", "charttime", "amount", "label")
            .collect()
        )

    def preprocess_labevents(self) -> pl.DataFrame:
        r"""Keep the laboratory measurements used by Bilos et al."""
        admission_ids = self.preprocess_admissions().select("hadm_id").lazy()
        labevents = (
            self.raw_dataset.load_table("labevents")
            .join(admission_ids, on="hadm_id", how="semi")
            .select("subject_id", "hadm_id", "charttime", "valuenum", "itemid")
            .join(
                self.raw_dataset.load_table("d_labitems").select("itemid", "label"),
                on="itemid",
                how="inner",
            )
            .with_columns(pl.col("label").cast(pl.String))
        )
        common_labels = (
            labevents.group_by("label")
            .agg(pl.col("subject_id").n_unique().alias("n_subjects"))
            .sort("n_subjects", descending=True, maintain_order=True)
            .head(150)
            .select("label")
        )
        return (
            labevents.join(common_labels, on="label", how="semi")
            .filter(pl.col("label").is_in(LAB_LABELS))
            .select("subject_id", "hadm_id", "charttime", "valuenum", "label")
            .collect()
        )

    def preprocess_outputevents(self) -> pl.DataFrame:
        r"""Keep the 15 selected output-event variables."""
        admission_ids = self.preprocess_admissions().select("hadm_id").lazy()
        return (
            self.raw_dataset.load_table("outputevents")
            .join(admission_ids, on="hadm_id", how="semi")
            .join(
                self.raw_dataset.load_table("d_items").select("itemid", "label"),
                on="itemid",
                how="inner",
            )
            .with_columns(pl.col("label").cast(pl.String))
            .filter(pl.col("label").is_in(OUTPUT_LABELS))
            .select("subject_id", "hadm_id", "charttime", "value", "label")
            .collect()
        )

    def preprocess_prescriptions(self) -> pl.DataFrame:
        r"""Apply the prescription filters from ``prescriptions.ipynb``."""
        admission_ids = self.preprocess_admissions().select("hadm_id").lazy()
        prescriptions = (
            self.raw_dataset.load_table("prescriptions")
            .join(admission_ids, on="hadm_id", how="semi")
            .with_columns(
                pl.col("drug").cast(pl.String),
                pl.col("dose_unit_rx").cast(pl.String),
            )
            .filter(pl.col("drug").is_in(PRESCRIPTION_LABELS))
            .filter(pl.col("dose_unit_rx").is_not_null())
            .with_columns(
                pl.when(
                    pl.col("drug").is_in(("D5W", "Sodium Chloride 0.9%  Flush"))
                    & pl.col("dose_unit_rx").eq("ml")
                )
                .then(pl.lit("mL"))
                .otherwise(pl.col("dose_unit_rx"))
                .alias("dose_unit_rx")
            )
        )
        prescriptions = _keep_matching_units(
            prescriptions,
            column="dose_unit_rx",
            label_column="drug",
            expected={
                "Acetaminophen": "mg",
                "D5W": "mL",
                "Heparin": "UNIT",
                "Insulin": "UNIT",
                "Magnesium Sulfate": "gm",
                "Potassium Chloride": "mEq",
                "Bisacodyl": "mg",
                "Pantoprazole": "mg",
            },
        )
        return prescriptions.select(
            "subject_id",
            "hadm_id",
            pl.col("starttime").alias("charttime"),
            pl.col("dose_val_rx").cast(pl.Float32, strict=False).alias("valuenum"),
            (pl.col("drug") + pl.lit(" Drug")).alias("label"),
        ).collect()

    def clean_raw_timeseries(self) -> pl.DataFrame:
        r"""Merge events into the wide frame exported as ``full_dataset.csv``."""
        inputs = self.preprocess_inputevents().rename({"amount": "valuenum"})
        laboratories = self.preprocess_labevents()
        outputs = self.preprocess_outputevents().rename({"value": "valuenum"})

        # ``datamerging.ipynb`` accidentally drops every prescription row via
        # ``presc_df.drop((presc_df["valuenum"] == "3-10").index)``: ``.index``
        # is the index of the whole Boolean series, rather than only matching
        # rows. See https://github.com/mbilos/neural-flows-experiments/issues/7.
        # The supplied export and ``label_dict.csv`` thus contain only the 102
        # input, laboratory, and output labels. Prescription start times must
        # likewise not contribute to the per-admission reference time, because
        # they can precede retained ICU events by several weeks. Keep
        # ``preprocess_prescriptions`` as the translation of that notebook, but
        # omit its data while reproducing the exported result.
        events = pl.concat(
            (inputs, laboratories, outputs), how="vertical_relaxed"
        ).lazy()
        reference_times = events.group_by("hadm_id").agg(
            pl.col("charttime").min().alias("reference_time")
        )
        feature_codes = pl.LazyFrame(
            {
                "label": LABEL_NAMES,
                "label_code": list(range(len(LABEL_NAMES))),
            },
            schema={"label": pl.String, "label_code": pl.UInt8},
        )
        events = (
            events.join(reference_times, on="hadm_id", how="inner")
            .with_columns(pl.col("label").cast(pl.String))
            .join(feature_codes, on="label", how="inner")
            .with_columns(
                (pl.col("charttime") - pl.col("reference_time"))
                .dt.total_minutes()
                .alias("time_stamp")
            )
            .filter(pl.col("time_stamp").lt(48 * 60))
            .with_columns(pl.col("time_stamp").cast(pl.Int16))
            .drop_nulls(["hadm_id", "time_stamp", "valuenum"])
        )
        value_columns = [f"Value_label_{label}" for label in range(102)]
        mask_columns = [f"Mask_label_{label}" for label in range(102)]
        wide = events.group_by("hadm_id", "time_stamp").agg(
            *(
                pl.when(pl.col("label_code").eq(label))
                .then(pl.col("valuenum"))
                .otherwise(0.0)
                .max()
                .cast(pl.Float32)
                .alias(value_column)
                for label, value_column in enumerate(value_columns)
            ),
            *(
                pl.when(pl.col("label_code").eq(label))
                .then(1)
                .otherwise(0)
                .max()
                .cast(pl.UInt8)
                .alias(mask_column)
                for label, mask_column in enumerate(mask_columns)
            ),
        )
        return (
            wide.collect()
            .select(*RAWDATA_SCHEMA)
            .sort("hadm_id", "time_stamp")
            .fill_nan(None)
        )

    def clean_timeseries(self) -> pl.DataFrame:
        r"""Apply the masking, normalization, and 5σ filtering from Bilos et al."""
        wide = self.raw_timeseries
        value_columns = [f"Value_label_{label}" for label in range(102)]
        mask_columns = [f"Mask_label_{label}" for label in range(102)]
        target_columns = [f"Value_{label}" for label in range(102)]
        masked = wide.select(
            pl.col("hadm_id").cast(pl.Int32),
            pl.col("time_stamp").cast(pl.Int16),
            *(
                pl.when(pl.col(mask_column).eq(1))
                .then(pl.col(value_column))
                .otherwise(None)
                .alias(target_column)
                for value_column, mask_column, target_column in zip(
                    value_columns, mask_columns, target_columns, strict=True
                )
            ),
        ).filter(
            pl.any_horizontal(pl.col(column).is_not_null() for column in target_columns)
        )
        normalized = masked.select(
            "hadm_id",
            "time_stamp",
            *(
                ((pl.col(column) - pl.col(column).mean()) / pl.col(column).std())
                .cast(pl.Float32)
                .alias(column)
                for column in target_columns
            ),
        )
        return (
            normalized.select(
                "hadm_id",
                "time_stamp",
                *(
                    pl.when(pl.col(column).is_between(-5, 5, closed="none"))
                    .then(pl.col(column))
                    .otherwise(None)
                    .cast(pl.Float32)
                    .alias(column)
                    for column in target_columns
                ),
            )
            .select(*TARGET_SCHEMA)
            .sort("hadm_id", "time_stamp")
        )

    def load_table(
        self, key: Literal["raw_timeseries", "timeseries"], /
    ) -> pl.DataFrame:
        r"""Load the materialized time-series table."""
        return pl.read_parquet(self.dataset_paths[key])


def _keep_matching_units(
    table: pl.LazyFrame,
    /,
    *,
    column: str,
    expected: dict[str, str],
    label_column: str = "label",
) -> pl.LazyFrame:
    r"""Discard entries for a selected label that use an unexpected unit."""
    return table.filter(
        *(
            pl.col(label_column).ne(label) | pl.col(column).eq(unit)
            for label, unit in expected.items()
        )
    )
