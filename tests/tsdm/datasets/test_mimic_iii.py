import gzip
from zipfile import ZipFile

import polars as pl
import pyarrow as pa

from tsdm.datasets import MIMIC_III, MIMIC_III_RAW, MIMIC_III_DeBrouwer2019


def test_mimic_iii_raw_preprocessing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MIMIC_III_RAW, "DATASET_ROOT_DIR", tmp_path)
    ds = MIMIC_III_RAW(initialize=False, verbose=False)
    schema = ds.get_schema("CALLOUT")
    values = dict.fromkeys(schema, "")
    values |= {
        "ROW_ID": "1",
        "REQUEST_TELE": "1",
        "REQUEST_RESP": "0",
        "REQUEST_CDIFF": "1",
        "REQUEST_MRSA": "0",
        "REQUEST_VRE": "1",
    }
    csv = ",".join(schema) + "\n" + ",".join(values.values()) + "\n"

    with ZipFile(ds.rawdata_paths[ds.rawdata_files[0]], "w") as archive:
        archive.writestr(ds.filelist["CALLOUT"], gzip.compress(csv.encode()))

    ds.clean("CALLOUT", validate_rawdata=False)
    table = ds.load_table("CALLOUT")

    assert isinstance(table, pl.LazyFrame)
    assert table.collect().select(
        "REQUEST_TELE",
        "REQUEST_RESP",
        "REQUEST_CDIFF",
        "REQUEST_MRSA",
        "REQUEST_VRE",
    ).row(0) == (1, 0, 1, 0, 1)

    monkeypatch.setattr(MIMIC_III, "DATASET_ROOT_DIR", tmp_path / "processed")
    processed = MIMIC_III(initialize=False, verbose=False)
    processed.clean("CALLOUT", validate_rawdata=False)
    processed_table = processed.load_table("CALLOUT")

    assert isinstance(processed_table, pa.Table)
    assert processed_table.select(
        [
            "REQUEST_TELE",
            "REQUEST_RESP",
            "REQUEST_CDIFF",
            "REQUEST_MRSA",
            "REQUEST_VRE",
        ]
    ).to_pylist() == [
        {
            "REQUEST_TELE": 1,
            "REQUEST_RESP": 0,
            "REQUEST_CDIFF": 1,
            "REQUEST_MRSA": 0,
            "REQUEST_VRE": 1,
        }
    ]


def test_mimic_iii_preprocessing() -> None:
    MIMIC_III_DeBrouwer2019.reset_dataset_files(force=True)
    ds = MIMIC_III_DeBrouwer2019()

    for key in MIMIC_III_DeBrouwer2019.table_names:
        assert isinstance(ds[key], pl.DataFrame)
        assert dict(ds[key].schema) == MIMIC_III_DeBrouwer2019.table_schemas[key]
