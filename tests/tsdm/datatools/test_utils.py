r"""Tests for generic data utilities."""

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch

from tsdm.datatools import get_schema, random_partition


def test_random_partition_dictionary_keys() -> None:
    r"""Partition dictionary keys into train, validation, and test datasets."""
    dataset = {f"sample-{index}": index for index in range(10)}
    train_keys, validation_keys, test_keys = random_partition(
        dataset.keys(), sizes=[4, 4, 2], rng=0
    )

    train_set = {key: dataset[key] for key in train_keys}
    validation_set = {key: dataset[key] for key in validation_keys}
    test_set = {key: dataset[key] for key in test_keys}

    assert tuple(map(len, (train_set, validation_set, test_set))) == (4, 4, 2)
    assert train_set.keys().isdisjoint(validation_set)
    assert train_set.keys().isdisjoint(test_set)
    assert validation_set.keys().isdisjoint(test_set)
    assert train_set.keys() | validation_set.keys() | test_set.keys() == dataset.keys()


def test_random_partition_tensor_and_labels() -> None:
    r"""Partition paired tensor features and string labels by shared indices."""
    features = torch.arange(30).reshape(10, 3)
    labels = [f"label-{index}" for index in range(len(features))]
    train_indices, validation_indices, test_indices = random_partition(
        range(len(features)), ratios=[0.4, 0.4, 0.2], rng=0
    )

    train_set = (features[train_indices], [labels[index] for index in train_indices])
    validation_set = (
        features[validation_indices],
        [labels[index] for index in validation_indices],
    )
    test_set = (features[test_indices], [labels[index] for index in test_indices])

    assert tuple(
        features.shape for features, _ in (train_set, validation_set, test_set)
    ) == ((4, 3), (4, 3), (2, 3))
    assert tuple(
        len(labels) for _, labels in (train_set, validation_set, test_set)
    ) == (4, 4, 2)
    assert all(
        torch.equal(row, features[int(label.removeprefix("label-"))])
        for dataset_features, dataset_labels in (train_set, validation_set, test_set)
        for row, label in zip(dataset_features, dataset_labels, strict=True)
    )


@pytest.mark.parametrize(
    ("table", "expected_columns", "expected_dtypes", "expected_index_columns"),
    [
        (
            pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=["left", "right"]),
            ["left", "right"],
            {"left": pd.Series([1]).dtype, "right": pd.Series([1]).dtype},
            [],
        ),
        (pd.Index([1, 2], name="id"), ["id"], {"id": pd.Series([1]).dtype}, []),
        (
            pd.Series([1, 2], name="value", index=pd.Index([3, 4], name="id")),
            ["value"],
            {"value": pd.Series([1]).dtype},
            ["id"],
        ),
        (
            pd.DataFrame({"value": [1, 2]}, index=pd.Index([3, 4], name="id")),
            ["value"],
            {"value": pd.Series([1]).dtype},
            ["id"],
        ),
        (
            pa.table({"value": [1, 2]}),
            ["value"],
            {"value": pa.int64()},
            [],
        ),
        (
            pl.DataFrame({"value": [1, 2]}),
            ["value"],
            {"value": pl.Int64},
            [],
        ),
    ],
)
def test_get_schema(
    table,
    expected_columns,
    expected_dtypes,
    expected_index_columns,
) -> None:
    r"""Extract schema information from each supported table type."""
    columns, dtypes, index_columns = get_schema(table)

    assert columns == expected_columns
    assert dtypes == expected_dtypes
    assert index_columns == expected_index_columns
