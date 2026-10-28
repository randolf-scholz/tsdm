r"""MIMIC-III processed version of Zhang et al. (ICML 2024).

References:
    - Irregular Multivariate Time Series Forecasting: \
      A Transformable Patching Graph Neural Networks Approach \
      Weijia Zhang, Chenlong Yin, Hao Liu, Xiaofang Zhou, Hui Xiong \
      Proceedings of the 41st International Conference on Machine Learning \
      <https://proceedings.mlr.press/v235/zhang24bw.html>
    - <https://github.com/usail-hkust/t-PatchGNN>
    - <https://physionet.org/content/mimic-iii-ext-tpatchgnn/1.0.0/>
"""

__all__ = ["MIMIC_III_Zhang2024"]

from functools import cached_property
from getpass import getpass
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import polars as pl
import torch

from tsdm.datasets.base import BaseDataset
from tsdm.utils import remote

type MIMIC_III_Zhang2024_Key = Literal[
    "LICENSE", "README", "SHA256SUMS", "full_dataset", "mimic", "variable_dict"
]


class MIMIC_III_Zhang2024(BaseDataset[MIMIC_III_Zhang2024_Key, object]):
    r"""Load the t-PatchGNN release without changing its data representation.

    This dataset is a curated subset of MIMIC-III (v1.4), specifically formatted to facilitate
    reproducibility of the experiments in the work t-PatchGNN. It serves as part of a benchmark designed
    for forecasting irregular multivariate clinical time series, that is, given a set of historical
    Irregular Multivariate Time Series (IMTS) observations and forecasting queries, the forecasting
    problem aims to accurately forecast the values in correspondence to these queries. This requires
    addressing key challenges such as missing data, variable sampling rates, and complex temporal
    dependencies. The dataset includes patient records with diverse physiological measurements, each
    sampled at irregular intervals, reflecting real-world clinical scenarios. It is structured to
    capture both short-term and long-term temporal patterns, making it well-suited for evaluating
    machine learning models in medical time series forecasting. By providing a standardized benchmark,
    this dataset aims to advance research in predictive modeling for healthcare, enabling the
    development of robust algorithms that can handle irregular and sparse clinical data. The dataset’s
    applications extend to critical areas such as early disease detection, patient risk stratification,
    and treatment outcome prediction, making it a valuable resource for the medical AI and machine
    learning communities.
    """

    DEFAULT_VERSION = "1.0.0"
    SOURCE_URL = r"https://physionet.org/content/mimic-iii-ext-tpatchgnn/get-zip"
    INFO_URL = r"https://physionet.org/content/mimic-iii-ext-tpatchgnn/"

    table_names = (
        "LICENSE",
        "README",
        "SHA256SUMS",
        "full_dataset",  # 23_457 patients
        "mimic",  # list[tuple[int, Float[$K], Float[$K, 96], Mask[$K, 96]]] (23_457)
        # ID, timestamps, values (sparse, unobserved=0.0), mask is float coded boolean.
        "variable_dict",
    )

    @cached_property
    def rawdata_files(self) -> list[str]:  # type: ignore
        r"""Name of the complete PhysioNet release archive."""
        return [f"mimic-iii-ext-tpatchgnn-{self.version}.zip"]

    @cached_property
    def filelist(self) -> dict[MIMIC_III_Zhang2024_Key, str]:
        r"""Map table names to members of the release archive."""
        prefix = f"mimic-iii-ext-tpatchgnn-{self.version}"
        return {
            "LICENSE": f"{prefix}/LICENSE.txt",
            "README": f"{prefix}/README.md",
            "SHA256SUMS": f"{prefix}/SHA256SUMS.txt",
            "full_dataset": f"{prefix}/full_dataset.csv.zip",
            "mimic": f"{prefix}/mimic.pt",
            "variable_dict": f"{prefix}/variable_dict.csv",
        }

    @cached_property
    def dataset_paths(self) -> dict[MIMIC_III_Zhang2024_Key, Path]:
        r"""Use PT for the Torch payload and Parquet for Polars tables."""
        return {
            key: self.DATASET_DIR / f"{key}.{'pt' if key == 'mimic' else 'parquet'}"
            for key in self.table_names
        }

    def get_rawdata_file(self, fname: str, /) -> None:
        r"""Download the release using a credentialed PhysioNet account."""
        remote.download(
            f"{self.SOURCE_URL}/{self.version}/",
            self.rawdata_paths[fname],
            username=input("PhysioNet username: "),
            password=getpass(prompt="PhysioNet password: "),
            headers={"User-Agent": "Wget/1.21.2"},
        )

    def clean_table(self, key: MIMIC_III_Zhang2024_Key, /) -> object:
        r"""Read release metadata or dispatch to the corresponding data loader."""
        match key:
            case "LICENSE" | "README" | "SHA256SUMS":
                with ZipFile(self.rawdata_paths[self.rawdata_files[0]]) as archive:
                    lines = (
                        archive.read(self.filelist[key]).decode("utf-8").splitlines()
                    )
                if key == "SHA256SUMS":
                    return pl.DataFrame(
                        [line.split(maxsplit=1) for line in lines if line.strip()],
                        schema={"value": pl.String, "filename": pl.String},
                        orient="row",
                    )
                return pl.DataFrame({"text": lines})
            case _:
                return super().clean_table(key)

    def clean_full_dataset(self) -> pl.DataFrame:
        r"""Read the CSV inside the nested ZIP without imposing a schema."""
        with (
            ZipFile(self.rawdata_paths[self.rawdata_files[0]]) as archive,
            BytesIO(archive.read(self.filelist["full_dataset"])) as buffer,
            ZipFile(buffer) as csv_archive,
        ):
            # The enclosing directory, if any, is not part of the table schema.
            members = [
                name
                for name in csv_archive.namelist()
                if Path(name).name == "full_dataset.csv"
            ]
            if len(members) != 1:
                raise ValueError(f"Expected one full_dataset.csv, found {members!r}.")
            with csv_archive.open(members[0]) as file:
                return pl.read_csv(file, infer_schema_length=None)

    def clean_variable_dict(self) -> pl.DataFrame:
        r"""Read the variable dictionary without imposing a schema."""
        with (
            ZipFile(self.rawdata_paths[self.rawdata_files[0]]) as archive,
            archive.open(self.filelist["variable_dict"]) as file,
        ):
            return pl.read_csv(file, infer_schema_length=None)

    def clean_mimic(self) -> object:
        r"""Load the serialized Torch payload on CPU, preserving its structure."""
        with (
            ZipFile(self.rawdata_paths[self.rawdata_files[0]]) as archive,
            archive.open(self.filelist["mimic"]) as file,
        ):
            return torch.load(file, map_location="cpu", weights_only=True)

    def store_table(self, key: MIMIC_III_Zhang2024_Key, table: object, /) -> None:
        r"""Persist the Torch payload or a Polars table."""
        if key == "mimic":
            torch.save(table, self.dataset_paths[key])
        elif isinstance(table, pl.DataFrame):
            table.write_parquet(self.dataset_paths[key])
        else:
            raise TypeError(f"Expected a Polars DataFrame for {key!r}.")

    def load_table(self, key: MIMIC_III_Zhang2024_Key, /) -> object:
        r"""Reload a processed table in its original in-memory representation."""
        if key == "mimic":
            return torch.load(
                self.dataset_paths[key], map_location="cpu", weights_only=True
            )
        return pl.read_parquet(self.dataset_paths[key])
