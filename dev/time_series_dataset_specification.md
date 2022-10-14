# Time Series Datasets

After working with several Time Series Datasets, we identified 3 main types of Time Series Datasets (2 fundamental ones and 1 subtype).

# Definition (Mathematical)

## Definition: Time Series Dataset (TSD)

A **time series dataset** $D$ is tuple $D=(S, M)$, consiting of

- Time Indexed Data $S = \{(t_i, v_i) ∣ i= 1:n\}$ is a set of tuples of **timestamps** $t∈𝓣$ and **values** $v∈𝓥$.
- Time independent **metadata** $M∈𝓜$.

## Definition: Time Series Collection (TSC)

A **time series collection** $C$ is triplet $C=(I, 𝓓, G)$ consiting of

- Index $I⊆𝓘$
- A set of time series datasets indexed by $I$, i.e. $𝓓=\{D_{i}∣i∈I\} = \{(S_i, M_i)∣i∈I\}$.
- "Outer"/"global" metadata $G∈𝓖$ that is independent of the index $I$. (Better name suggestion appreciated)

Note that here, each time series might be specified over a different space with different observed values and metadata, i.e. $S_i = \{ (t_j^{(i)}), v_{j}^{(i)}  ∣ j = 1…n_i\}$ with $t_j^{(i)}∈𝓣_i$ and $v_j^{(i)}∈𝓥_i$ and $M_i∈𝓜_i$. So for example $D_1$ might contain an audio recording whereas $D_2$ contains price data of a specific stock.

### Variant: _equimodal_ Time Series Collection

If all time series in a TSC have the same data modality, i.e. they share the same kind of time index, metadata types and value types: $𝓜_i=𝓜$, $𝓣_i=𝓣$ and $𝓥_i=𝓥$ for all $i∈I$, then we say that the TSC is **equimodal**.

For the purpose of the KIWI project, it seems appropriate to only work with equimodal TSCs. Applying meta-learning / multi-task-learning Machine Learning over non-equimodal TSCs is in principle possible, but much more complicated.

# Implementation Details

With regard to the implementation, we make the following simplifying assumptions:

1. We assume all **values** can be represented in simple vectorial form. If the data contains non-vectorial information, such as images, or spectral data, we need to think about adding additional tables. In this case, [`xarray`](https://docs.xarray.dev/en/stable/) datasets may offer a useful abstraction.
2. We assume the metadata can be represented as 3 tables of interest:
   1. `metadata`: Time-independent information
   2. `timeseries_features`: Time Series Channel description (better name suggestion appreciated)
   3. `metadata_features`: Metadata Column description (better name suggestion appreciated)
3. Additionally, we require each TSD to contain a `tmin` and a `tmax` field, which indicate the beginning and end of the time series. All time stamps in the dataset must be between `tmin` and `tmax` (both included).
   - By default, these can be initialized with the smallest/largest timestamp in the dataset.
   - Design Question: Make these part of the metadata or not?

**Remark:** 2.2 and 2.3 can be superfluous if the database allows annotation/metadata per columns, or datatypes with units.

**Remark:** For most ML applications, global metadata is superfluous, it would only become relevant for meta-learning across collections of TSC (i.e. sort of nested collections of time series).

## Definition (Programming) - simplified

## Implementation (TimeSeriesDataset)

```python
@dataclass
class TimeSeriesDataset:
    timeseries:          DataFrame                   # index: Time
    metadata:            Optional[DataFrame] = None  # single row!
    timeseries_features: Optional[DataFrame] = None  # rows = timeseries.columns
    metadata_features:   Optional[DataFrame] = None  # rows = metadata.columns
    # tmin/tmax possible mandatory columns in metadata?
    tmin:                TimeLike                    # smallest timestamp
    tmax:                TimeLike                    # largest  timestamp
```

```python
@dataclass
class TimeSeriesCollection:  # equimodal !
    index:               Index                       # contains IDs
    timeseries:          DataFrame                   # multi-index: [ID, Time]
    metadata:            Optional[DataFrame] = None  # index: ID
    timeseries_features: Optional[DataFrame] = None  # rows = timeseries.columns
    metadata_features:   Optional[DataFrame] = None  # rows = metadata.columns
    global_metadata:     Optional[DataFrame] = None
    # tmin/tmax possible mandatory columns in metadata?
    tmin:                Series[TimeLike]            # index: ID
    tmax:                Series[TimeLike]            # index: ID
```

# Examples

## Example: Clinical Patient Records (MIMIC-III / MIMIC-IV datasets)

These records contain clinical patient data. For each patient admitted to the hospital/ICU, several time series are recorded such as blood pressure, heart rate, etc.

- Index: patient_id / admission_id
- Time Series: Online measurements such as heart rate, blood pressure, etc.
- metadata: age/sex/gender/weight/preconditions per patient
- global metadata: hospital name
- timeseries_features: unit of measurement (e.g. heart-rate: bpm, blood pressure: mmHg),
- metadata_features: unit of measurement, e.g. height: inches, weight: pounds, etc.

## Example: BioProcess Data (iLab TU Berlin)

- Index: tuple (run_id, experiment_id)
- Time Series: time-dependent measurements of sensors: DOT, acetate, glucose, etc., as well as _covariates_ (controls): stirring speed, temperature, etc.
- metadata: bioreactor used, bioreactor volume, organism used, etc.
- global metadata: lab equipment details
- timeseries_features: unit of measurement, measurement error (device specific), biologically plausible ranges for values, etc.
- metadata_features: unit of measurement (e.g. reactor volume: mL), etc.

## Complicated Example: Video Data

A video contains several kinds of time-indexed data: images, sound and possibly subtitles.
E.g. in this case: $$𝓥 = \underbrace{ℝ^{m×n}}_{\text{image}} ⊕ \underbrace{ℝ^{k}}_{\text{audio}} ⊕ \underbrace{\operatorname{Seq}(\text{ASCII})}_{\text{text}}$$

# Remarks on DataTypes

## Nullable DataTypes

Time Series often contain missing values. We **STRONGLY** suggest to make use of the nullable DataTypes, such as `pandas.BooleanDType`, `pandas.Int64DType` and `pandas.Float64DType` or `pyarrow` Datatypes.

## Categorical Columns

Columns that don't contain numerical values often are simply encoded as strings. However, often it is appropriate to make use of **categorical datatypes** instead (such as `pandas.CategoricalDtype` or `pyarrow.DictionaryType`). This has the following crucial advantages:

1. Saves a lot of memory, since internally it stores the column as small integer type plus a lookup dictionary.
2. Avoids duplicate entries ("Acetate" vs "acetate"), since the user can search the list or existing categories, and oinly if necessary create a new one. From time to time, duplicate categories that emerged in the data anyway should be merged.
3. Allows Null-values. Otherwise, missing values need to be encoded with strings such as `""`, `"-"`, `"nan"`, `"N/A"`, `"NA"`, `"NIL"` or `"NULL"`, etc. again with lots of spelling variants leading to duplicate entries.
4. Is required for ML classification models anyway.

Only when string data is appropriate should it really be stored as strings, such as in `comments` or `descriptions` fields, but not in principally categorical fields such as `organism_name`, `reactor_name`, etc.

## Time-Like DataTypes

Multiple datatypes are useful for encoding timestamps. However, crucially a distinction needs to be made between
timestamp datatypes, which only store a point in time, and timedelta datatypes, which encode a time duration.

We say a datatype `S` is _`TimeDeltaLike`_, if it encodes a time duration, not a specific point in time. It must satisfy the following properties:

1. `S` is [totally ordered](https://en.wikipedia.org/wiki/Total_order)
2. `S` is an additive Group (might be relaxed to additive monoid)
3. The ordering is compatible with the addition, i.e. `S` is in fact a [linearly/totally ordered group](https://en.wikipedia.org/wiki/Linearly_ordered_group)

We say a `TimeDeltaLike` datatype encodes **continuous** time if it supports a scalar multiplication with floats, otherwise we say it encodes **discrete** time.

We say a datatype `T` is _`TimeStampLike`_ if it encodes a specific point in time. It must satisfy the following properties:

1. It is [totally ordered](https://en.wikipedia.org/wiki/Total_order)
2. It supports a subtraction operation `T×T→S` where `S` is some `TimeDeltaLike` type.
   `S` may or may not be equal to `T`. For example, if `T=np.datetime64` then `S=np.timedelta64`, but if `T=float` then `S=float`

We specify the following types as TimeStamp / TimeDelta-like. All TSCs should use one of these types, or a user defined type that satisfies the properties above.

If a TSD uses a `datetime` as the time index, it is appropriate to store `tmin` and `tmax` in the metadata.

```python
# NOTE: Might be more appropriate to implement as typing.Protocol.
TimeStampLike: TypeAlias = int | uint | float | datetime
TimeDeltaLike: TypeAlias = int | float | timedelta
TimeLike: TypeAlias = TimeStampLike | TimeDeltaLike

# Continuous Time Variants
ContinuousTimeStampLike: TypeAlias = float | datetime
ContinuousTimeDeltaLike: TypeAlias = float | timedelta
ContinuousTimeLike: TypeAlias = ContinuousTimeStampLike | ContinuousTimeDeltaLike

# Discrete Time Variants
DiscreteTimeStampLike: TypeAlias = int | uint
DiscreteTimeDeltaLike: TypeAlias = int
DiscreteTimeLike: TypeAlias = ContinuousTimeStampLike | ContinuousTimeDeltaLike
```

# Data Export

We suggest export / storage as `parquet` files using `pyarrow`. It is very fast and supports nullable data types.
See <https://arrow.apache.org/docs/python/index.html>. It has one major downside: The format is not binary stable due to batched storage. Thus, `sha256` hashes may differ when the same data is stored on different machines.
