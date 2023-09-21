# Time Series Datasets v 1.1

After working with several Time Series Datasets, we identified 3 main types of Time Series Datasets (2 fundamental ones and 1 subtype).

## Definition (Mathematical)

### Definition: Time Series Dataset (TSD)

A **time series dataset** $D$ is tuple $D=(S, M)$, consiting of

- Time Indexed Data $S = \{(t_i, v_i) ∣ i= 1:n\}$ is a set of tuples of **timestamps** $t∈𝓣$ and **values** $v∈𝓥$.
- Time independent **metadata** $M∈𝓜$.

### Definition: Time Series Collection (TSC)

A **time series collection** $C$ is triplet $C=(I, 𝓓, G)$ consiting of

- Index $I⊆𝓘$
- A set of time series datasets indexed by $I$, i.e. $𝓓=\{D_{i}∣i∈I\} = \{(S_i, M_i)∣i∈I\}$.
- "Outer"/"global" metadata $G∈𝓖$ that is independent of the index $I$. (Better name suggestion appreciated)

Note that here, each time series might be specified over a different space with different observed values and metadata, i.e. $S_i = \{ (t_j^{(i)}), v_{j}^{(i)}  ∣ j = 1…n_i\}$ with $t_j^{(i)}∈𝓣_i$ and $v_j^{(i)}∈𝓥_i$ and $M_i∈𝓜_i$. So for example $D_1$ might contain an audio recording whereas $D_2$ contains price data of a specific stock.

#### Variant: _equimodal_ Time Series Collection

If all time series in a TSC have the same data modality, i.e. they share the same kind of time index, metadata types and value types: $𝓜_i=𝓜$, $𝓣_i=𝓣$ and $𝓥_i=𝓥$ for all $i∈I$, then we say that the TSC is **equimodal**.

For the purpose of the KIWI project, it seems appropriate to only work with equimodal TSCs. Applying meta-learning / multi-task-learning Machine Learning over non-equimodal TSCs is in principle possible, but much more complicated.

#### Variant: _multi-table_ Time Series

In some cases, it might not be sensible to encode the time series data in a single tensor.
For example, for video data, one might have time-indexed image and audio data, the former being naturally 2-dimensional (or even 3 dimensional when considering color channels) and the latter being 1-dimensional.

Of course, in principle the data could be flattened into a single tensor (in particular, under the hood the computer does this as memory is a 1-dimensional array), but this is not very convenient for the user.

So, we say that a time series dataset is **single-table** if the data is expressed as an element of a tensor product space of "elementary spaces", e.g. $𝓥 = 𝓥^{n_1} ⊗ 𝓥^{n_2} ⊗ … ⊗ 𝓥^{n_k}$. Otherwise, if it is a direct sum of some spaces, i.e. $𝓥 = 𝓥_1 ⊕ 𝓥_2 ⊕ … ⊕ 𝓥_k$, then we say that the time series dataset is **multi-table**.

The same applies for the metadata.

## Implementation Details

For simplicity, we will only consider **single-table** time series datasets (i.e. all observations at timestamp $t$ can be naturally stored in a single n-dimensional tensor. This would not include cases like video data, where one might have a 2-dimensional image and a 1-dimensional audio signal at each timestamp that need to be stored in separate tables if one wishes to avoid having to flatten the image data).

### Implementation: Time Series Dataset (TSD)

Following the mathematical definition, we require the following data to specifiy a time-series dataset:

```yaml
TimeSeriesDataset: # tuple[timeseries, metadata, time_features, value_features, metadata_features]
  timeseries:  # DataFrame                      # index: Time, alternative: tuple[TimeArray, ValueArray]
    time:      Array[Time, dtype=TimeLike]    # sorted ascending, indexes the other tables
    values:    Array[shape=(len(time), ...)]
  metadata:  Optional[DataFrame] = None       # single row DataFrame / dict
    ...                                       # arbitrary metadata
  # space descriptors
  time_features: # Optional[DataFrame]         # single row DataFrame / dict
    unit:  category[str]  # The unit of the time index (e.g. "s" or "h" if TimeArray is int/float)
    scale: category[str]  # The scale "percent", "absolute", "linear", "logarithmic", etc.
    dtype: category[str]  # The data type of the time index (e.g. "int" or "float")
    lower: TimeLike       # The time when the time series starts (must be ≤ t[0])
    upper: TimeLike       # The time when the time series ends (must be ≥ t[-1])
    # ...                   # Other time features (e.g. error bounds if clocks imprecise, etc.)
  value_features: # Optional[DataFrame] = None  # rows = timeseries.columns
    unit:  category[str]  # The unit of the values (e.g. "m" or "°C")
    scale: category[str]  # The scale "percent", "absolute", "linear", "logarithmic", etc.
    dtype: category[str]  # The data type of the values (e.g. "float32" or "int32")
    lower: numerical      # Lower Bound of the values, if applicable
    upper: numerical      # Upper Bound of the values, if applicable
    # ...                   # Other timeseries features (e.g. error bounds, etc.)
  metadata_features: # Optional[DataFrame] = None  # rows = metadata.columns
    unit:  category[str]  # The unit of the values (e.g. "m" or "°C")
    scale: category[str]  # The scale "percent", "absolute", "linear", "logarithmic", etc.
    dtype: category[str]  # The data type of the values (e.g. "float32" or "int32")
    lower: numerical      # Lower Bound of the values, if applicable
    upper: numerical      # Upper Bound of the values, if applicable
    # ...                   # Other metadata features (e.g. error bounds, etc.)
```

The optional tables `time_features`, `timeseries_features` and `metadata_features` encode information describing the spaces $𝓣$, $𝓥$ and $𝓜$, respectively. This information

For a **multi-table** timeseries dataset such as video data, we might have 3 tables `image`, `audio` and `subtitles` instead of just a single `values` table.

```yaml
timeseries:
  time: Array[Time] # sorted: ascending.
  # indexes the other tables, i.e. they have 1 row per timestamp
  image: Array[Sample, Height, Width, Channels, dtype=int8]
  audio: Array[Sample, Channel, dtype=int8]
  subtitles: Array[Sample, Language, dtype=string]
```

### Implementation: Time Series Collection (TSC)

There are 2 main differences:

- Additional table `index`
- The `timeseries` table is now a dictionary of `timeseries` tables, one for each time series in the collection.
- The `metadata` table is now a dictionary of `metadata` tables, one for each time series in the collection.
- The space descriptors are now also tables, with `time_features[i]`, `value_features[i]` and `metadata_features[i]` describing the spaces $𝓣ᵢ$, $𝓥ᵢ$ and $𝓜ᵢ$, respectively.
- Additional table `index_features` describing the space $𝓘$.
- Additional tables `global_metadata` that is independent of the index

```yaml
TimeSeriesCollection: # generic
  index: Required[Array]
  data: Required[dict[index, TSD]]
  global_metadata: Optional[Array]
  # space descriptors
  index_features: Optional[Array]
  global_features: Optional[Array]
```

However, we are mostly interested in the **equimodal** case, i.e. when all the TSDs share the same value and metadata space $𝓣ᵢ=𝓣$, $𝓥ᵢ = 𝓥$ and $𝓜ᵢ = 𝓜$ for all $i∈𝓘$. We do however allow differences in the upper/lower boundary values. So `time_features`, `value_features` and `metadata_features` each be either of `None`, a simple table or a multi-index table using $𝓘$ as the first level.

In practice a convenient solution is to use `timedelta64` format for the time-index which in most cases alleviates the need for any time-features besides lower and upper bounds.

```yaml
TimeSeriesCollection: # equimodal
  index: Required[Array]
  timeseries: Required[Array[index, time, values]]
  metadata: Optional[Array[index, metadata]]
  global_metadata: Optional[Array]
  # space descriptors
  index_features: Optional[Array[unit, scale, dtype, lower, upper, ...]]
  time_features: Optional[Array[index, unit, scale, dtype, lower, upper, ...]]
  values_features: Optional[Array[index, unit, scale, dtype, lower, upper, ...]]
  metadata_features: Optional[Array[index, unit, scale, dtype, lower, upper, ...]]
  global_features: Optional[Array[unit, scale, dtype, lower, upper, ...]]
```

## Examples

### Example: Clinical Patient Records (MIMIC-III / MIMIC-IV datasets)

These records contain clinical patient data. For each patient admitted to the hospital/ICU, several time series are recorded such as blood pressure, heart rate, etc.

- Index: patient_id / admission_id
- Time Series: Online measurements such as heart rate, blood pressure, etc.
- metadata: age/sex/gender/weight/preconditions per patient
- global metadata: hospital name
- timeseries_features: unit of measurement (e.g. heart-rate: bpm, blood pressure: mmHg),
- metadata_features: unit of measurement, e.g. height: inches, weight: pounds, etc.

### Example: BioProcess Data (iLab TU Berlin)

- Index: tuple (run_id, experiment_id)
- Time Series: time-dependent measurements of sensors: DOT, acetate, glucose, etc., as well as _covariates_ (controls): stirring speed, temperature, etc.
- metadata: bioreactor used, bioreactor volume, organism used, etc.
- global metadata: lab equipment details
- timeseries_features: unit of measurement, measurement error (device specific), biologically plausible ranges for values, etc.
- metadata_features: unit of measurement (e.g. reactor volume: mL), etc.

### Complicated Example: Video Data

A video contains several kinds of time-indexed data: images, sound and possibly subtitles.
E.g. in this case: $$𝓥 = \underbrace{ℝ^{m×n}}_{\text{image}} ⊕ \underbrace{ℝ^{k}}_{\text{audio}} ⊕ \underbrace{\operatorname{Seq}(\text{ASCII})}_{\text{text}}$$

## Remarks on DataTypes

### Nullable DataTypes

Time Series often contain missing values. We **STRONGLY** suggest to make use of the nullable DataTypes, such as `pandas.BooleanDType`, `pandas.Int64DType` and `pandas.Float64DType` or `pyarrow` Datatypes.

### Categorical Columns

Columns that don't contain numerical values often are simply encoded as strings. However, often it is appropriate to make use of **categorical datatypes** instead (such as `pandas.CategoricalDtype` or `pyarrow.DictionaryType`). This has the following crucial advantages:

1. Saves a lot of memory, since internally it stores the column as small integer type plus a lookup dictionary.
2. Avoids duplicate entries ("Acetate" vs "acetate"), since the user can search the list or existing categories, and oinly if necessary create a new one. From time to time, duplicate categories that emerged in the data anyway should be merged.
3. Allows Null-values. Otherwise, missing values need to be encoded with strings such as `""`, `"-"`, `"nan"`, `"N/A"`, `"NA"`, `"NIL"` or `"NULL"`, etc. again with lots of spelling variants leading to duplicate entries.
4. Is required for ML classification models anyway.

Only when string data is appropriate should it really be stored as strings, such as in `comments` or `descriptions` fields, but not in principally categorical fields such as `organism_name`, `reactor_name`, etc.

### Time-Like DataTypes

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
from typing import TypeAlias
from datetime import datetime, timedelta
# NOTE: Might be more appropriate to implement as typing.Protocol.
TimeStampLike: TypeAlias = int | float | datetime
TimeDeltaLike: TypeAlias = int | float | timedelta
TimeLike: TypeAlias = TimeStampLike | TimeDeltaLike

# Continuous Time Variants
ContinuousTimeStampLike: TypeAlias = float | datetime
ContinuousTimeDeltaLike: TypeAlias = float | timedelta
ContinuousTimeLike: TypeAlias = ContinuousTimeStampLike | ContinuousTimeDeltaLike

# Discrete Time Variants
DiscreteTimeStampLike: TypeAlias = int
DiscreteTimeDeltaLike: TypeAlias = int
DiscreteTimeLike: TypeAlias = ContinuousTimeStampLike | ContinuousTimeDeltaLike
```

## Data Export

We suggest export / storage as `parquet` files using `pyarrow`. It is very fast and supports nullable data types.
See <https://arrow.apache.org/docs/python/index.html>. It has one major downside: The format is not binary stable due to batched storage. Thus, `sha256` hashes may differ when the same data is stored on different machines.
