# TOTOs

## Multi-Datasets

A single dataset can have multiple processed versions originating from the same raw data.

- For example, a dataset can have a version with and without the `NaN` values removed.
- A dataset might be available in multiple versions (e.g. MIMIC-IV v1.0 and v2.0).

We want to allow subclasses of a `BaseDataset`, which work as follows:

- They share the same `RAWDATADIR`.
  - This stores the raw data, potentially multiple versions like `mimic-iv-v1.0.zip` and `mimic-iv-v2.0.zip`
  - The `DATSETDIR` contains a base folder for the dataset, e.g. `MIMIC-IV` with potentially multiple subfolders.
- The `BaseDataset` class provides
  - Downloading functionality.
  - Light preprocessing, in particular conversion of the data to binary parquet format with appropriate column types.
  - "Subclasses" can use this data to build more complex preprocessing pipelines.

Questions:

- Do "subclasses" really make sense here?
  - It seems sufficient that the dataset would instantiate the "base class".
- The download functionality etc. should be maintained.

Implementation:

Create 3 subclass of BaseDataset:

1. `MIMIC_IV` the base_class
2. A `DerivedDataset` abstract base class that is parametrized by a `BaseDataset`
3. The subclass / instance of the `DerivedDataset` class.

```python
class MIMIC_IV(BaseDataset):
    ...


class DerivedDataset(ABCMeta):
    ...


class MIMIC_IV_DeBrouwer(
    DerivedDataset[MIMIC_IV["1.0"]]
):  # <-- too magic for type checkers / linters
    RAWDATADIR = MIMIC_IV.RAWDATADIR
    DATASETDIR = MIMIC_IV.DATASETDIR / MIMIC_IV_DeBrouwer
    RAWDATAFILES = MIMIC_IV.RAWDATAFILES
    RAWDATAPATHS = MIMIC_IV.RAWDATAPATHS
    download = MIMIC_IV.download


@extends(MIMIC_IV)
class MIMIC_IV_DeBrouwer(FrameDataset, ...):
    ...
```

Question: How to combine with Versioning? We want to allow multiple versions of the same dataset. Options:

1. MIMIC_IV @ "1.0"
2. MIMIC_IV["1.0"] => returns a class `"MIMIC-IV@1.0"` that can then be instantiated.
3. MIMIC_IV(version="1.0")

Should different versions be different classes, or different instances of the same class?
At least in the case of MIMIC-IV, the data, e.g. the columns in the table may change between versions.
Thus, it seems versions should be different classes.

=> Create `MetaClass` called `VersionedDataset`?

```python
class DerivedDatasetMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any) -> None:
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if os.environ.get("GENERATING_DOCS", False):
            cls.RAWDATA_DIR = Path(f"~/.tsdm/rawdata/{cls.__name__}/")
            cls.DATASET_DIR = Path(f"~/.tsdm/datasets/{cls.__name__}/")
        else:
            cls.RAWDATA_DIR = RAWDATADIR / cls.__name__
            cls.DATASET_DIR = DATASETDIR / cls.__name__

    def __getitem__(cls, klass: type[BaseDataset]) -> type[BaseDataset]:
        r"""Get the dataset class."""
        return klass
```

---

### OLD TODOs

- Rezero as parameterization
- Rezero class
- Generic classes
- StandardScalar is not properly vectorized. Do custom implementation instead.
- Exact loss function? Forecast?
- Without Standard-scaling
- RMSE **NOT** channel wise, but flattened
- accumulate cross-validation: RMSE
- horizon 1h / 2h / 6h ↭ avg number of timestamps
- horizons: 150+100, 300+300, 400+600
- In TS split: first snippet that fits until last snippet
- Time Encoding?
- negative values: truncate to zero!

timeseries.min()

    Flow_Air                           0.000000
    StirringSpeed                      0.000000
    Temperature                       32.689999
    Acetate                           -0.257518
    Base                               0.000000
    Cumulated_feed_volume_glucose      3.000000
    Cumulated_feed_volume_medium       5.882958
    DOT                                0.000000
    Glucose                           -0.094553
    OD600                             -0.962500
    Probe_Volume                     200.000000
    pH                                 0.000000
    Fluo_GFP                        -250.000000
    InducerConcentration               0.000000
    Volume                            -2.235569

Current WorkFLow

KIWI data - cut off negative values

- push tsdm with KIWI task
- run linodenet on KIWI data
- create graphs
- encode metadata
- results with metadata
- results on Electricity, ETTh, Traffic

ADD DATASETS:

- ECL: <https://drive.google.com/file/d/1rUPdR7R2iWFW-LMoDdHoO2g4KgnkpFzP/view?usp=sharing>
- Weather: <https://drive.google.com/file/d/1UBRz-aM_57i_KCC-iaSWoKDPTGGv6EaG/view?usp=sharing>
