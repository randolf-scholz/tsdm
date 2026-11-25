r"""Regression tests for filled Arrow arrays."""

from typing import Any

import pyarrow as pa
import pytest

from tsdm.backend.pyarrow import full_like, null_like


@pytest.fixture(params=["array", "chunked", "empty", "empty_chunked"])
def arr(request: pytest.FixtureRequest) -> pa.Array | pa.ChunkedArray:
    r"""Include invalid numeric strings, nulls, and empty chunks."""
    match request.param:
        case "array":
            return pa.array(["not a number", None, "text"])
        case "chunked":
            return pa.chunked_array([["not a number"], [], [None, "text"]])
        case "empty":
            return pa.array([], type=pa.string())
        case "empty_chunked":
            return pa.chunked_array([], type=pa.string())
        case _:
            raise ValueError(f"Unexpected array kind: {request.param}")


@pytest.mark.parametrize(
    "fill_value",
    [
        pa.scalar("filled"),
        pa.scalar(7, type=pa.int16()),
        pa.scalar(None, type=pa.float32()),
        pa.scalar([1, 2]),
        pa.scalar({"x": 1}),
        pa.NA,
        "python string",
        42,
        None,
    ],
)
def test_full_like(arr: pa.Array | pa.ChunkedArray, *, fill_value: Any) -> None:
    r"""Fill every position without casting or retaining the input values."""
    original = arr.to_pylist()
    value = fill_value if isinstance(fill_value, pa.Scalar) else pa.scalar(fill_value)
    dtype = arr.type if pa.types.is_null(value.type) else value.type

    result = full_like(arr, fill_value=fill_value)

    assert result.type == dtype
    assert result.to_pylist() == [value.as_py()] * len(arr)
    assert arr.to_pylist() == original
    assert isinstance(result, pa.ChunkedArray) == isinstance(arr, pa.ChunkedArray)
    if isinstance(arr, pa.ChunkedArray):
        actual_sizes = [len(chunk) for chunk in result.chunks]
        expected_sizes = [len(chunk) for chunk in arr.chunks]
        assert actual_sizes == expected_sizes


@pytest.mark.parametrize(
    "values",
    [
        pa.array([1, None, 3], type=pa.int16()),
        pa.array([True, None, False]),
        pa.array(["text", None, "other"]),
        pa.array([[1, 2], None, []]),
        pa.array([{"x": 1}, None, {"x": 2}]),
        pa.array(["text", None, "other"]).dictionary_encode(),
        pa.nulls(3),
        pa.array([], type=pa.int64()),
    ],
)
@pytest.mark.parametrize("chunked", [False, True])
def test_null_like(values: pa.Array, *, chunked: bool) -> None:
    r"""Return only nulls while preserving dtype and the container kind."""
    arr = pa.chunked_array([values[:1], values[1:]]) if chunked else values
    original = arr.to_pylist()

    result = null_like(arr)

    assert result.type == arr.type
    assert result.to_pylist() == [None] * len(arr)
    assert result.null_count == len(arr)
    assert arr.to_pylist() == original
    assert isinstance(result, pa.ChunkedArray) == chunked
    if chunked:
        actual_sizes = [len(chunk) for chunk in result.chunks]
        expected_sizes = [len(chunk) for chunk in arr.chunks]
        assert actual_sizes == expected_sizes
