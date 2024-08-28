# Comparison with other libraries

1. [`darts`](https://github.com/unit8co/darts) is a Python library for user-friendly forecasting and anomaly detection on time series.
2. [`aeon`](https://github.com/aeon-toolkit/aeon) is an open-source toolkit for learning from time series.
3. [`pytorch-forecasting`](https://github.com/jdb78/pytorch-forecasting) is a PyTorch-based package for forecasting time series with state-of-the-art network architectures.
4. [`tsai`](https://github.com/timeseriesAI/tsai) is an open-source deep learning package built on top of Pytorch & fastai focused on state-of-the-art techniques for time series tasks like classification, regression, forecasting, imputation…
5. [`TSLib`](https://github.com/thuml/Time-Series-Library) is an open-source library for deep learning researchers, especially for deep time series analysis.
6. [`sktime`](https://github.com/sktime/sktime) is a library for time series analysis in Python.
7. [`tslearn`](https://github.com/tslearn-team/tslearn) is a machine learning toolkit for time series analysis in Python.

## Overview

|                                     | `tsdm` | `aeon` | `darts` | `pytorch-forecasting` | `sktime` | `tsai` | `tslearn` | `tslib` |
|-------------------------------------|--------|--------|---------|-----------------------|----------|--------|-----------|---------|
| **typing**                          |        |        |         |                       |          |        |           |         |
| typed (coverage)                    |        |        |         |                       |          |        |           |         |
| modern type-hints (PEP 585/604/695) |        |        |         |                       |          |        |           |         |
| mypy/pyright                        |        |        |         |                       |          |        |           |         |
| **features**                        |        |        |         |                       |          |        |           |         |
| continuous time                     |        |        |         |                       |          |        |           |         |
| duplicate values                    |        |        |         |                       |          |        |           |         |
| missing values (basic/universal)    |        |        |         |                       |          |        |           |         |
| static covariates                   |        |        |         |                       |          |        |           |         |
| dynamic covariates                  |        |        |         |                       |          |        |           |         |
| **infrastructure**                  |        |        |         |                       |          |        |           |         |
| datasets                            |        |        |         |                       |          |        |           |         |
| datasets-ext                        |        |        |         |                       |          |        |           |         |
| encoders                            |        |        |         |                       |          |        |           |         |
| encoders-ext                        |        |        |         |                       |          |        |           |         |
| models                              |        |        |         |                       |          |        |           |         |
| models-ext                          |        |        |         |                       |          |        |           |         |
| tasks                               |        |        |         |                       |          |        |           |         |
| tasks-ext                           |        |        |         |                       |          |        |           |         |
| **performance**                     |        |        |         |                       |          |        |           |         |
| **extensibility**                   |        |        |         |                       |          |        |           |         |
| Abstract Base Classes               |        |        |         |                       |          |        |           |         |
| Protocols                           |        |        |         |                       |          |        |           |         |
| Wrappers                            |        |        |         |                       |          |        |           |         |
| **usability**                       |        |        |         |                       |          |        |           |         |
| **documentation**                   |        |        |         |                       |          |        |           |         |

- typed:
  - we ran both mypy and pyright on the codebase.
  - we ran `pyright --verifytypes` on the codebase to get the type coverage.
  - we test with `ruff check select=UP` if modern type hints are used.

## Comparison with Darts

Darts offers the `darts.TimeSeries` class as a way to encapsulate time series.
However, we have abserved that this class does not work well with highly irregular time series data.
For instance, trying to convert one of the time series from the `KiwiBenchmark` dataset (experiment `439/15325`) to a `darts.TimeSeries` requires to setting a frequency.
However, as the time series is highly irregular, we have to set a very low frequency of 1 second, and the `DataFrame` blows up from a size of 1529 rows with a memory usage of less than 100KB to over 50k rows with over 6MB of memory usage.

```python
import tsdm
import darts
ds = tsdm.datasets.KiwiBenchmark()
ts = ds.timeseries.loc[439, 15325]
sc = ds.static_covariates.loc[439, 15325]
ts.info()

# convert to float as darts is incompatible with pyarrow backend
ts_compat = ts.astype(float)
# reset index
ts_compat=ts_compat.reset_index()
# convert index as darts is incompatible with time delta index
ts_compat=ts_compat.assign(
    time=ts_compat.pop("elapsed_time").astype("timedelta64[s]") + sc["start_time"]
)
# convert to darts.TimeSeries
ts_darts = darts.TimeSeries.from_dataframe(ts_compat, time_col="time", freq="1s")
```
