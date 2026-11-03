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

__all__ = ["MIMIC_III_Zhang2024", "MIMIC_III_Record", "SCHEMAS"]

from functools import cached_property
from getpass import getpass
from io import BytesIO
from pathlib import Path
from typing import Literal, NamedTuple, cast, overload
from zipfile import ZipFile

import polars as pl
import torch

from tsdm.datasets.base import BaseDataset
from tsdm.pprint import pprint_repr
from tsdm.utils import remote

type MIMIC_III_Zhang2024_Key = Literal[
    "LICENSE", "README", "SHA256SUMS", "full_dataset", "mimic", "variable_dict"
]


@pprint_repr
class MIMIC_III_Record(NamedTuple):
    r"""One patient's observations, with a variable number K of timestamps.

    ``timestamps`` has shape (K,); ``values`` and ``mask`` have shape (K, 96).
    Timestamps and values retain their source floating dtype. Masks are Boolean;
    unobserved values retain their source zero encoding.

    The dataset saves a dictionary of tensors, IDs, and lengths, reconstructing
    named records as views on loading. The PT file needs only PyTorch to load.
    """

    id: int
    timestamps: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor


SCHEMAS: dict[MIMIC_III_Zhang2024_Key, dict[str, pl.DataType]] = {
    "SHA256SUMS": {"value": pl.String(), "filename": pl.String()},
    "full_dataset": {
        "ID": pl.Int64(),
        "Time": pl.Float64(),
        **{
            name: dtype
            for k in range(96)
            for name, dtype in (
                (f"Value_label_{k}", pl.Float32()),
                (f"Mask_label_{k}", pl.Boolean()),
            )
        },
    },
    "variable_dict": {"LABEL": pl.String(), "LABEL_CODE": pl.Int8()},
}


class MIMIC_III_Zhang2024(BaseDataset[MIMIC_III_Zhang2024_Key, object]):
    r"""Load the t-PatchGNN release with typed values and Boolean masks.

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

    LICENSE: str
    README: str
    SHA256SUMS: pl.DataFrame
    full_dataset: pl.DataFrame
    mimic: list[MIMIC_III_Record]
    variable_dict: pl.DataFrame

    table_names = (
        "LICENSE",
        "README",
        "SHA256SUMS",
        "full_dataset",  # 23_457 patients, ID, Time, value/mask columns for labels 0..95
        "mimic",  # list[MIMIC_III_Record], with Boolean masks and unobserved values=0.0
        "variable_dict",
    )
    table_schemas = SCHEMAS

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
        r"""Use text files for documents, PT for records, and Parquet for tables."""
        formats = {"LICENSE": "txt", "README": "md", "mimic": "pt"}
        return {
            key: self.DATASET_DIR / f"{key}.{formats.get(key, 'parquet')}"
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
        r"""Read release metadata or dispatch to the corresponding cleaning helper."""
        match key:
            case "LICENSE" | "README" | "SHA256SUMS":
                with ZipFile(self.rawdata_paths[self.rawdata_files[0]]) as archive:
                    text = archive.read(self.filelist[key]).decode("utf-8")
                if key == "SHA256SUMS":
                    return pl.DataFrame(
                        [
                            line.split(maxsplit=1)
                            for line in text.splitlines()
                            if line.strip()
                        ],
                        schema={"value": pl.String, "filename": pl.String},
                        orient="row",
                    )
                return text
            case "full_dataset":
                return self.clean_full_dataset()
            case "variable_dict":
                return self.clean_variable_dict()
            case "mimic":
                return self.clean_mimic()
            case _:
                raise KeyError(f"Key {key} unknown! Must be one of {self.table_names}")

    def clean_full_dataset(self) -> pl.DataFrame:
        r"""Read Float32 values and Boolean masks, preserving CSV column order."""
        # CSV masks contain 0/1, which Polars cannot parse directly as Boolean.
        raw_schema = {
            name: pl.Int8() if dtype == pl.Boolean else dtype
            for name, dtype in self.table_schemas["full_dataset"].items()
        }
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
                return pl.read_csv(
                    file, schema_overrides=raw_schema, infer_schema=False
                ).with_columns(pl.col("^Mask_label_.*$").cast(pl.Boolean))

    def clean_variable_dict(self) -> pl.DataFrame:
        r"""Read labels with Int8 codes, excluding the redundant CSV index."""
        with (
            ZipFile(self.rawdata_paths[self.rawdata_files[0]]) as archive,
            archive.open(self.filelist["variable_dict"]) as file,
        ):
            return pl.read_csv(
                file,
                columns=["LABEL", "LABEL_CODE"],
                schema_overrides=self.table_schemas["variable_dict"],
                infer_schema=False,
            )

    def clean_mimic(self) -> list[MIMIC_III_Record]:
        r"""Load named patient records on CPU and convert masks to Boolean."""
        with (
            ZipFile(self.rawdata_paths[self.rawdata_files[0]]) as archive,
            archive.open(self.filelist["mimic"]) as file,
        ):
            records = torch.load(file, map_location="cpu", weights_only=True)
        return [
            MIMIC_III_Record(patient_id, timestamps, values, mask.bool())
            for patient_id, timestamps, values, mask in records
        ]

    def store_table(self, key: MIMIC_III_Zhang2024_Key, table: object, /) -> None:
        r"""Store documents as text and records as a portable packed dictionary.

        The PT dictionary contains ``ids`` and ``lengths`` lists plus concatenated
        ``timestamps``, ``values``, and Boolean ``mask`` tensors. Patient boundaries
        are given by ``lengths``. This avoids serializing three tensors per patient.
        """
        if key in {"LICENSE", "README"}:
            self.dataset_paths[key].write_text(cast("str", table), encoding="utf-8")
        elif key == "mimic":
            records = cast("list[MIMIC_III_Record]", table)
            packed = {
                "ids": [record.id for record in records],
                "lengths": [record.timestamps.numel() for record in records],
                "timestamps": torch.cat([record.timestamps for record in records])
                if records
                else torch.empty(0),
                "values": torch.cat([record.values for record in records])
                if records
                else torch.empty((0, 96)),
                "mask": torch.cat([record.mask.bool() for record in records])
                if records
                else torch.empty((0, 96), dtype=torch.bool),
            }
            torch.save(packed, self.dataset_paths[key])
        elif isinstance(table, pl.DataFrame):
            table.write_parquet(self.dataset_paths[key])
        else:
            raise TypeError(f"Expected a Polars DataFrame for {key!r}.")

    @overload
    def load_table(self, key: Literal["mimic"], /) -> list[MIMIC_III_Record]: ...
    @overload
    def load_table(self, key: Literal["LICENSE", "README"], /) -> str: ...
    @overload
    def load_table(
        self, key: Literal["SHA256SUMS", "full_dataset", "variable_dict"], /
    ) -> pl.DataFrame: ...
    def load_table(self, key: MIMIC_III_Zhang2024_Key, /) -> object:
        r"""Load documents, tables, or patient views over memory-mapped tensors.

        Older list-based PT files remain readable, but must be reprocessed to use
        packed storage. Mapping defers tensor I/O until values are accessed;
        records use disjoint views into the three concatenated tensors.
        """
        if key in {"LICENSE", "README"}:
            return self.dataset_paths[key].read_text(encoding="utf-8")
        if key == "mimic":
            data = torch.load(
                self.dataset_paths[key],
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            if isinstance(data, dict):
                lengths = data["lengths"]
                records = zip(
                    data["ids"],
                    data["timestamps"].split(lengths),
                    data["values"].split(lengths),
                    data["mask"].split(lengths),
                    strict=True,
                )
            else:
                records = data
            return [
                MIMIC_III_Record(patient_id, timestamps, values, mask.bool())
                for patient_id, timestamps, values, mask in records
            ]
        return pl.read_parquet(self.dataset_paths[key])
