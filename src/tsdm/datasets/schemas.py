r"""Schemas for Datasets Metadata."""

__all__ = ["DEFAULT_METADATA_SCHEMA"]

DEFAULT_METADATA_SCHEMA = {
    "variable"        : "string[pyarrow]",
    "dtype"           : "string[pyarrow]",
    "lower_bound"     : "float64[pyarrow]",
    "upper_bound"     : "float64[pyarrow]",
    "lower_inclusive" : "bool[pyarrow]",
    "upper_inclusive" : "bool[pyarrow]",
    "unit"            : "string[pyarrow]",
    "description"     : "string[pyarrow]",
}  # fmt: skip
