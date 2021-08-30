datasets
========

Mental Model h5 files
---------------------

All datasets are stored in `Hierarchical Data Format <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_, based on

This format knows two things:

- Datasets, which are multidimensional arrays of a homogeneous type
- Groups, which are container structures which can hold datasets and other groups

For example, pandas will store a single DataFrame with different data type columns as
multiple Series objects of homogeneous data-type, collected in a group.


Supported DataTypes:

For Datasets: Only nullable types.

Dtypes:

- BooleanDtype
- CategoricalDtype
- DatetimeTZDtype
- Float32Dtype
- Float64Dtype
- Int16Dtype
- Int32Dtype
- Int64Dtype
- Int8Dtype
- IntervalDtype
- PeriodDtype
- SparseDtype
- StringDtype
- UInt16Dtype
- UInt32Dtype
- UInt64Dtype
- UInt8Dtype

Index Types:

- CategoricalIndex
- DatetimeIndex
- Float64Index
- Index
- IndexSlice
- Int64Index
- IntervalIndex
- MultiIndex
- PeriodIndex
- RangeIndex
- TimedeltaIndex
- UInt64Index

Arrays:

- ArrowStringArray
- BooleanArray
- Categorical
- DatetimeArray
- FloatingArray
- IntegerArray
- IntervalArray
- PandasArray
- PeriodArray
- SparseArray
- StringArray
- TimedeltaArray


