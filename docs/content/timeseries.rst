Working with Time Series Data
=============================

The module `tsdm.data.timeseries` provides the classes `TimeSeries` and
`TimeSeriesCollection`, which are the main data structures for wrapping time
series data.

A `TimeSeries` consists of the following components:

1. `timeseries`: a table indexed by a time-like index (datetime, timedelta,
   float or integer) with one or more columns representing the time series data.
2. `static_covariates`: An optional vector of data that has no time stamp associated
   with it. This data is assumed to be constant over the entire time series.
3. `metadata_timeseries`: An optional table that describes the columns of the
   time series. This table can be used to store additional information about the
   time series data, such as the units of the data, the source of the data, etc.
3. `metadata_static_covariates`: An optional table that describes the columns of the
   static covariates. This table can be used to store additional information about the
   static covariates, such as the units of the data, the source of the data, etc.

A `TimeSeriesCollection` is a collection of `TimeSeries` objects, grouped by an outer index.
That is, it behaves like a dictionary of `TimeSeries` objects, where the keys are the outer index.
