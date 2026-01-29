r"""Type Aliases/Protocols for time series."""

__all__ = [
    "TS_Keys",
    "TSC_Keys",
]


from typing import Literal

type TS_Keys = Literal["timeseries", "timeseries_metadata"]
r"""Type Alias for time series dataset keys."""
type TSC_Keys = Literal[
    "timeseries",
    "timeseries_metadata",
    "static_covariates",
    "static_covariates_metadata",
]
r"""Type Alias for time series collections dataset keys."""
