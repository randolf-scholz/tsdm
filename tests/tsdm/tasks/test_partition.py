r"""Shared partition contracts and random/rotating allocation policies."""

from collections.abc import Callable
from functools import partial
from math import nextafter
from random import Random, getstate
from types import MappingProxyType
from typing import Any, ClassVar

import pytest
import torch

from tsdm.stats.distances import total_variation_distance
from tsdm.tasks.base import SplitType
from tsdm.tasks.utils import is_partition, random_partitions, rotating_partitions


class TestPartitions:
    r"""Common contracts inherited by the concrete partition test classes."""

    __test__ = False
    function: ClassVar[Callable[..., list]]
    parameter_name: ClassVar[str]

    @pytest.fixture
    def fold_function(self) -> Callable[..., list]:
        r"""Return the partition implementation under test."""
        return self.function

    @pytest.fixture
    def parameter(self) -> str:
        r"""Name the implementation's relative allocation argument."""
        return self.parameter_name

    @pytest.fixture
    def fold_generator(self) -> Callable[..., list[dict[str, list]]]:
        r"""Configure two equal named splits with a reproducible local seed."""
        return partial(
            self.function,
            **{self.parameter_name: {"train": 1, "test": 1}},
            num=2,
            rng=12,
        )

    @pytest.fixture(params=[1, 2])
    def ratio_partitioner(self, request: pytest.FixtureRequest) -> Callable[..., list]:
        r"""Exercise relative allocation and error bounds for one or two draws."""
        return partial(
            self.function, **{self.parameter_name: [1, 1]}, num=request.param, rng=0
        )

    @pytest.fixture
    def partitioner(self, request: pytest.FixtureRequest) -> Callable[..., list]:
        r"""Configure enough items for distinct draws and RNG-state comparisons."""
        return partial(
            request.getfixturevalue("fold_function"),
            range(20),
            **{self.parameter_name: {"train": 3, "test": 1}},
            num=4,
        )

    def test_error_check_disabled_by_default(
        self,
        ratio_partitioner: Callable[..., object],
    ) -> None:
        r"""Default and explicit None both permit large discrepancies without checking."""
        result = ratio_partitioner(range(1))
        assert result
        assert result == ratio_partitioner(range(1), max_error=None)

    def test_max_error_rejects_rounding(
        self,
        ratio_partitioner: Callable[..., object],
    ) -> None:
        r"""One occurrence cannot meet equal targets within an explicit five percent."""
        with pytest.raises(
            RuntimeError, match=r"total variation distance 0.5.*max_error=0.05"
        ):
            ratio_partitioner(range(1), max_error=0.05)

    def test_within_max_error(self, ratio_partitioner: Callable[..., object]) -> None:
        r"""Normalized proportions allow small rounding errors, despite unequal counts."""
        assert ratio_partitioner(range(21), max_error=0.05)

    def test_max_error_boundary(self, ratio_partitioner: Callable[..., object]) -> None:
        r"""A threshold equal to TV is accepted; a smaller threshold rejects the result."""
        assert ratio_partitioner(range(1), max_error=0.5)
        with pytest.raises(RuntimeError, match="total variation distance"):
            ratio_partitioner(range(1), max_error=0.499)

    def test_zero_max_error(self, ratio_partitioner: Callable[..., object]) -> None:
        assert ratio_partitioner(range(8), max_error=0)
        with pytest.raises(RuntimeError, match="max_error=0"):
            ratio_partitioner(range(1), max_error=0)

    def test_empty_inputs_without_error_check(
        self,
        ratio_partitioner: Callable[..., object],
    ) -> None:
        r"""Disabling the check returns empty splits without evaluating a distribution."""
        result = ratio_partitioner([])
        assert result in ([([], [])], [([], []), ([], [])])
        assert result == ratio_partitioner([], max_error=None)

    @pytest.mark.parametrize("max_error", [0.0, 1.0])
    def test_empty_inputs_have_no_distribution(
        self,
        ratio_partitioner: Callable[..., object],
        max_error: float,
    ) -> None:
        r"""The distance function rejects zero total weight for empty allocations."""
        with pytest.raises(ValueError, match="positive total weight"):
            ratio_partitioner([], max_error=max_error)

    @pytest.mark.parametrize(
        "max_error", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")]
    )
    def test_invalid_max_error(
        self, ratio_partitioner: Callable[..., object], max_error: float
    ) -> None:
        with pytest.raises(ValueError, match="max_error"):
            ratio_partitioner(range(8), max_error=max_error)

    def test_seed_and_instance_equivalence(
        self, partitioner: Callable[..., object]
    ) -> None:
        before = getstate()
        assert partitioner(rng=37) == partitioner(rng=Random(37))
        assert getstate() == before

    def test_existing_rng_advances_and_can_be_replayed(
        self,
        partitioner: Callable[..., object],
    ) -> None:
        rng = Random(37)
        initial = rng.getstate()
        first = partitioner(rng=rng)
        after_first = rng.getstate()
        second = partitioner(rng=rng)
        assert initial != after_first != rng.getstate()
        assert first != second

        rng.setstate(initial)
        assert partitioner(rng=rng) == first
        assert partitioner(rng=rng) == second

    @pytest.mark.parametrize("rng", [1.5, "37"])
    def test_invalid_random_source(
        self, partitioner: Callable[..., object], rng
    ) -> None:
        with pytest.raises(TypeError, match="rng"):
            partitioner(rng=rng)

    @pytest.mark.parametrize("num", [1, 4])
    @pytest.mark.parametrize("named", [True, False])
    def test_computed_threshold_is_compared_strictly(
        self, fold_function, parameter: str, num: int, *, named: bool
    ) -> None:
        r"""The computed distance passes, but even the next smaller threshold fails."""
        options: dict[str, Any] = {"num": num}
        spec = {"train": 3, "test": 1} if named else [3, 1]
        result = fold_function(range(10), **{parameter: spec}, **options, rng=0)
        partitions = result if isinstance(result, list) else [result]
        counts = [
            [
                len(split)
                for split in (part.values() if isinstance(part, dict) else part)
            ]
            for part in partitions
        ]
        error = total_variation_distance(counts, [3, 1]).max().item()
        assert fold_function(
            range(10), **{parameter: spec}, **options, rng=0, max_error=error
        )
        with pytest.raises(RuntimeError, match="total variation distance"):
            fold_function(
                range(10),
                **{parameter: spec},
                **options,
                rng=0,
                max_error=nextafter(error, 0),
            )

    @pytest.mark.parametrize("group_labels", [None, [0, 0, 1, 1, 2, 2, 3, 3]])
    def test_named_and_unnamed_folds(
        self, fold_function, parameter: str, *, group_labels
    ) -> None:
        r"""Both forms have identical assignments, with exact keys and empty splits."""
        marker = object()
        spec = MappingProxyType(
            {
                "train": 1,
                "training": 1,
                SplitType.VALIDATION: 1,
                ("custom", 1): 1,
                marker: 0,
                17: 0,
                None: 0,
            }
        )
        items = [[index] for index in range(8)]
        named = fold_function(
            items, **{parameter: spec}, num=2, rng=12, group_labels=group_labels
        )
        assert isinstance(named, list)
        assert len(named) == 2
        for partition in named:
            assert isinstance(partition, dict)
            assert all(
                actual is key for actual, key in zip(partition, spec, strict=True)
            )
            assert partition[marker] == partition[17] == partition[None] == []
            assert sorted(
                item[0] for part in partition.values() for item in part
            ) == list(range(8))
        for counts in (list(spec.values()), tuple(spec.values())):
            unnamed = fold_function(
                items, **{parameter: counts}, num=2, rng=12, group_labels=group_labels
            )
            assert isinstance(unnamed, list)
            assert unnamed == [tuple(partition.values()) for partition in named]
            assert all(isinstance(partition, tuple) for partition in unnamed)
            for partition in unnamed:
                assert len(partition) == len(counts)

    def test_unnamed_zero_weight_splits(self, fold_function, parameter: str) -> None:
        r"""Zero-weight splits remain empty, with independently mutable item lists."""
        folds = fold_function([1, 2], **{parameter: [2, 0]}, num=2, rng=0)
        assert folds == [([1, 2], []), ([1, 2], [])]
        folds[0][0].append(3)
        assert folds == [([1, 2, 3], []), ([1, 2], [])]

    @pytest.mark.parametrize("named", [True, False])
    def test_group_error_measures_items(
        self, fold_function, parameter: str, *, named: bool
    ) -> None:
        r"""Balanced group counts do not conceal highly imbalanced item counts."""
        spec = {"train": 1, "test": 1} if named else [1, 1]
        partitioner = partial(
            fold_function,
            range(10),
            **{parameter: spec},
            group_labels=[0] * 9 + [1],
            num=2,
            rng=0,
        )
        assert partitioner() == partitioner(max_error=None)
        with pytest.raises(RuntimeError, match=r"total variation distance 0.4"):
            partitioner(max_error=0.05)

    def test_group_error_aggregates_all_splits(
        self, fold_function, parameter: str
    ) -> None:
        r"""Four splits have TV 0.25 even though each proportion differs by only 0.125."""
        with pytest.raises(RuntimeError, match=r"total variation distance 0.25"):
            fold_function(
                range(8),
                **{parameter: [1, 1, 1, 1]},
                group_labels=[0, 0, 0, 1, 1, 1, 2, 3],
                num=2,
                rng=0,
                max_error=0.1875,
            )

    @pytest.mark.parametrize("group_labels", [None, [0, 1, 0, 2, 0, 3, 2, 3]])
    def test_unhashable_occurrences_match_index_partitions(
        self, fold_generator: Callable[..., list[dict[str, list]]], group_labels
    ) -> None:
        r"""Equal unhashable items retain identity, multiplicity, order, and allocation."""
        items = [[index % 2] for index in range(8)]
        items[-1] = items[0]  # Repeated references are separate occurrences too.
        original = items.copy()
        folds = fold_generator(items, group_labels=group_labels)
        expected = fold_generator(range(len(items)), group_labels=group_labels)
        assert folds == fold_generator(items, group_labels=group_labels)
        assert items == original
        for actual_fold, expected_fold in zip(folds, expected, strict=True):
            for key, indices in expected_fold.items():
                actual = actual_fold[key]
                assert len(actual) == len(indices)
                assert all(
                    item is items[index]
                    for item, index in zip(actual, indices, strict=True)
                )

    def test_duplicate_items_split_independently(
        self,
        fold_generator: Callable[..., list[dict[str, list]]],
    ) -> None:
        r"""Equal items split independently unless equal labels explicitly group them."""
        items = [1, 1, 1, 1]
        folds = fold_generator(items)
        assert all(part == [1, 1] for fold in folds for part in fold.values())

    def test_equal_items_can_be_used_as_labels(
        self,
        fold_generator: Callable[..., list[dict[str, list]]],
    ) -> None:
        r"""Passing items as labels keeps equal values together without losing copies."""
        items = [1, 2, 1, 2, 1, 2]
        folds = fold_generator(items, group_labels=items)
        for fold in range(2):
            train, test = folds[fold]["train"], folds[fold]["test"]
            assert sorted(train + test) == sorted(items)
            assert set(train).isdisjoint(test)
            assert sorted([train, test]) == [[1, 1, 1], [2, 2, 2]]

    @pytest.mark.parametrize("group_labels", [[], ["a", "b"]])
    def test_invalid_group_lengths(
        self, fold_generator: Callable[..., list[dict[str, list]]], group_labels
    ) -> None:
        with pytest.raises(ValueError, match="group_labels"):
            fold_generator([1], group_labels=group_labels)

    def test_unordered_labels_are_rejected(
        self,
        fold_generator: Callable[..., list[dict[str, list]]],
    ) -> None:
        with pytest.raises(TypeError, match=r"group_labels.*order"):
            fold_generator([1, 2], group_labels={"a", "b"})

    def test_unhashable_labels_are_rejected(
        self,
        fold_generator: Callable[..., list[dict[str, list]]],
    ) -> None:
        with pytest.raises(TypeError, match="unhashable"):
            fold_generator([1, 2], group_labels=[[1], [2]])

    def test_empty_group_labels(
        self,
        fold_generator: Callable[..., list[dict[str, list]]],
    ) -> None:
        assert fold_generator([], group_labels=[]) == fold_generator([])

    def test_group_integrity_and_reproducibility(self, fold_generator) -> None:
        r"""Unequal replicate groups stay intact with positionally aligned labels."""
        keys = [(1, 10), (2, 20), (1, 11), (3, 30), (1, 12), (4, 40)]
        group_labels = [(key[0], "red") for key in keys]
        group_by_key = dict(zip(keys, group_labels, strict=True))
        original_keys, original_groups = keys.copy(), group_labels.copy()
        before = getstate()
        folds = fold_generator(keys, group_labels=group_labels)
        again = fold_generator(tuple(keys), group_labels=tuple(group_labels))
        assert folds == again
        assert getstate() == before
        assert keys == original_keys
        assert group_labels == original_groups
        for fold in range(2):
            train, test = folds[fold]["train"], folds[fold]["test"]
            assert set(train).isdisjoint(test)
            assert set(train) | set(test) == set(keys)
            assert {group_by_key[key] for key in train}.isdisjoint(
                group_by_key[key] for key in test
            )
            assert len({group_by_key[key] for key in train}) == 2
            assert train == [key for key in keys if key in train]
            assert test == [key for key in keys if key in test]

    @pytest.mark.parametrize(
        "configuration", [{"train": 1, "training": 1}, {"unknown": 1}]
    )
    def test_arbitrary_split_names(
        self, configuration, fold_function, parameter: str
    ) -> None:
        r"""Former aliases and unknown names are independent, unchanged split keys."""
        items = range(sum(configuration.values()))
        partitions = fold_function(items, **{parameter: configuration}, num=1, rng=0)
        assert list(partitions[0]) == list(configuration)

    @pytest.mark.parametrize("num", [None, 0, -1, 1.5])
    def test_invalid_counts(self, num, fold_function, parameter: str) -> None:
        with pytest.raises(ValueError, match="num"):
            fold_function([], num=num, **{parameter: [2]}, rng=0)

    def test_num_is_required(self, fold_function, parameter: str) -> None:
        with pytest.raises(TypeError, match="num"):
            fold_function(range(4), **{parameter: [1, 1]}, rng=0)

    def test_unordered_identifiers_are_rejected(
        self, fold_function, parameter: str
    ) -> None:
        with pytest.raises(TypeError, match="order"):
            fold_function({1, 2}, num=1, **{parameter: [1]}, rng=0)


class TestRandomPartitions(TestPartitions):
    r"""Random draws, largest-remainder allocation, and exact group sizes."""

    __test__ = True
    function = staticmethod(random_partitions)
    parameter_name = "ratios"

    @pytest.fixture(params=["single_sizes", "ratios", "sizes"])
    def partitioner(self, request: pytest.FixtureRequest) -> Callable[..., list]:
        r"""Exercise both allocation modes and single/batched RNG lifetimes."""
        match request.param:
            case "single_sizes":
                return partial(random_partitions, range(20), num=1, sizes=(12, 8))
            case "ratios":
                return partial(
                    random_partitions, range(20), num=4, ratios={"train": 3, "test": 1}
                )
            case "sizes":
                return partial(
                    random_partitions, range(20), num=4, sizes={"train": 15, "test": 5}
                )
            case _:
                raise AssertionError(request.param)

    def test_random_source_is_required(
        self, partitioner: Callable[..., object]
    ) -> None:
        with pytest.raises(TypeError, match="rng"):
            partitioner()

    def test_none_random_source_is_invalid(
        self, partitioner: Callable[..., object]
    ) -> None:
        with pytest.raises(TypeError, match="rng"):
            partitioner(rng=None)

    @pytest.mark.parametrize("named", [True, False])
    @pytest.mark.parametrize("num", [1, 2])
    def test_explicit_sizes_skip_error_check(self, num: int, *, named: bool) -> None:
        sizes = {"train": 9, "test": 1} if named else [9, 1]
        partitions = random_partitions(
            range(10), num=num, sizes=sizes, rng=0, max_error=0
        )
        assert len(partitions) == num
        for partition in partitions:
            parts = partition.values() if isinstance(partition, dict) else partition
            assert [len(part) for part in parts] == [9, 1]

    def test_exact_sizes_and_tuple_identifiers(self) -> None:
        keys = [(index, "series") for index in range(7000)]
        parts = random_partitions(keys, num=1, sizes=(5000, 1000, 1000), rng=42)[0]
        assert tuple(len(part) for part in parts) == (5000, 1000, 1000)
        assert is_partition(parts, union=keys)
        assert (
            parts == random_partitions(keys, num=1, sizes=(5000, 1000, 1000), rng=42)[0]
        )

    def test_largest_remainder_ties_favor_earlier_parts(self) -> None:
        parts = random_partitions(range(1), num=1, ratios=(1, 1), rng=0)[0]
        assert tuple(len(part) for part in parts) == (1, 0)
        parts = random_partitions(range(5), num=1, ratios=(1, 1, 1), rng=0)[0]
        assert tuple(len(part) for part in parts) == (2, 2, 1)

    def test_large_finite_ratios(self) -> None:
        parts = random_partitions(range(5), num=1, ratios=(1e308, 1e308), rng=0)[0]
        assert tuple(len(part) for part in parts) == (3, 2)
        assert is_partition(parts, union=range(5))

    def test_empty_partitions(self) -> None:
        assert random_partitions([], num=1, ratios=(1, 0), rng=0)[0] == ([], [])
        assert random_partitions([], num=1, ratios={"train": 1, "test": 0}, rng=0)[
            0
        ] == {
            "train": [],
            "test": [],
        }

    @pytest.mark.parametrize("sizes", [(), (0, 0)])
    @pytest.mark.parametrize("named", [True, False])
    @pytest.mark.parametrize("num", [1, 2])
    def test_sizes_require_positive_count(
        self, sizes, num: int, *, named: bool
    ) -> None:
        with pytest.raises(ValueError, match=r"sizes.*positive"):
            random_partitions(
                [], num=num, sizes=dict(enumerate(sizes)) if named else sizes, rng=0
            )

    @pytest.mark.parametrize("sizes", [(-1, 3), (1, 0), (1.5, 0.5)])
    @pytest.mark.parametrize("named", [True, False])
    def test_invalid_sizes(self, sizes, *, named: bool) -> None:
        with pytest.raises(ValueError, match="sizes"):
            random_partitions(
                range(2), num=1, sizes=dict(enumerate(sizes)) if named else sizes, rng=0
            )[0]

    @pytest.mark.parametrize(
        "ratios", [(), (0, 0), (-1, 2), (float("nan"), 1), (float("inf"), 1)]
    )
    @pytest.mark.parametrize("named", [True, False])
    def test_invalid_ratios(self, ratios, *, named: bool) -> None:
        with pytest.raises(ValueError, match="ratios"):
            random_partitions(
                range(2),
                num=1,
                ratios=dict(enumerate(ratios)) if named else ratios,
                rng=0,
            )[0]

    @pytest.mark.parametrize("num", [1, 2])
    def test_exactly_one_allocation_mode(self, num: int) -> None:
        with pytest.raises(ValueError, match="Exactly one"):
            random_partitions([1], num=num, rng=0)  # type: ignore[call-overload]
        with pytest.raises(ValueError, match="Exactly one"):
            random_partitions([1], num=num, sizes=(1,), ratios=(1,), rng=0)  # type: ignore[call-overload]

    @pytest.mark.parametrize("parameter", ["sizes", "ratios"])
    def test_named_and_unnamed_partitions(self, parameter: str) -> None:
        r"""Mapping keys retain identity and order without changing seeded assignments."""
        marker = object()
        spec = MappingProxyType(
            {"train": 3, "training": 3, ("custom", 1): 2, marker: 0}
        )
        items = [[index % 2] for index in range(8)]
        named = (
            random_partitions(items, num=1, sizes=spec, rng=42)[0]
            if parameter == "sizes"
            else random_partitions(items, num=1, ratios=spec, rng=42)[0]
        )
        assert isinstance(named, dict)
        assert all(actual is key for actual, key in zip(named, spec, strict=True))
        assert [len(part) for part in named.values()] == [3, 3, 2, 0]
        for counts in (list(spec.values()), tuple(spec.values())):
            unnamed = (
                random_partitions(items, num=1, sizes=counts, rng=42)[0]
                if parameter == "sizes"
                else random_partitions(items, num=1, ratios=counts, rng=42)[0]
            )
            assert isinstance(unnamed, tuple)
            assert tuple(named.values()) == unnamed
            for named_part, unnamed_part in zip(named.values(), unnamed, strict=True):
                assert all(
                    left is right
                    for left, right in zip(named_part, unnamed_part, strict=True)
                )

    @pytest.mark.parametrize("parameter", ["sizes", "ratios"])
    @pytest.mark.parametrize("named", [True, False])
    def test_group_allocation(self, parameter: str, *, named: bool) -> None:
        r"""Group counts determine assignments regardless of the number of samples."""
        labels = [(1, "red"), (2, "red"), (1, "red"), (3, "blue"), (1, "red")]
        items = [[index] for index in range(len(labels))]
        items[-1] = items[0]
        unique_labels = list(dict.fromkeys(labels))
        spec = {"train": 1, ("test", 0): 2, "empty": 0} if named else [1, 2, 0]
        options: dict[str, Any] = {parameter: spec}
        parts = random_partitions(items, num=1, **options, group_labels=labels, rng=42)[
            0
        ]
        expected = random_partitions(unique_labels, num=1, **options, rng=42)[0]
        actual_parts = list(parts.values()) if isinstance(parts, dict) else parts
        expected_parts = (
            list(expected.values()) if isinstance(expected, dict) else expected
        )
        if named:
            assert list(parts) == list(spec)
        for actual, expected_labels in zip(actual_parts, expected_parts, strict=True):
            expected_items = [
                item
                for item, item_label in zip(items, labels, strict=True)
                if item_label in expected_labels
            ]
            assert len(actual) == len(expected_items)
            assert all(
                left is right
                for left, right in zip(actual, expected_items, strict=True)
            )
        assert len(actual_parts[-1]) == 0
        assert (
            parts
            == random_partitions(items, num=1, **options, group_labels=labels, rng=42)[
                0
            ]
        )

    @pytest.mark.parametrize("named", [True, False])
    @pytest.mark.parametrize("grouped", [True, False])
    @pytest.mark.parametrize("parameter", ["sizes", "ratios"])
    def test_batched_and_successive_partitions_agree(
        self, parameter: str, *, named: bool, grouped: bool
    ) -> None:
        r"""Successive draws agree on rounding, membership, input order, and RNG state."""
        items = [7, 0, 6, 1, 5, 2, 4, 3]
        labels = [0, 1, 0, 2, 3, 1, 4, 4] if grouped else None
        counts = [2, 2, 1, 0] if grouped else [3, 3, 2, 0]
        values = counts if parameter == "sizes" else [1, 1, 1, 0]
        spec = (
            dict(zip(["train", "valid", "test", "empty"], values, strict=True))
            if named
            else values
        )
        options: dict[str, Any] = {parameter: spec}
        rng = Random(37)
        reference_rng = Random(37)
        expected = random_partitions(
            items, num=3, **options, group_labels=labels, rng=reference_rng
        )
        actual = [
            random_partitions(items, num=1, **options, group_labels=labels, rng=rng)[0]
            for _ in range(3)
        ]
        assert actual == expected
        assert rng.getstate() == reference_rng.getstate()
        for partition in actual:
            parts = partition.values() if isinstance(partition, dict) else partition
            for part in parts:
                assert part == [item for item in items if item in part]

    @pytest.mark.parametrize("parameter", ["sizes", "ratios"])
    def test_unique_labels_match_ungrouped_partition(self, parameter: str) -> None:
        r"""Each occurrence is its own group by default, including duplicate values."""
        items = [[0], [1], [0], [1]]
        options: dict[str, Any] = {parameter: [2, 2]}
        assert (
            random_partitions(items, num=1, **options, rng=37)[0]
            == random_partitions(
                items, num=1, **options, group_labels=["d", "b", "a", "c"], rng=37
            )[0]
        )

    def test_group_sizes_count_groups(self) -> None:
        r"""Explicit sizes count groups, never sample counts."""
        assert random_partitions(
            range(4), num=1, sizes=[1, 0], group_labels=["a"] * 4, rng=0, max_error=0
        )[0] == ([0, 1, 2, 3], [])
        with pytest.raises(ValueError, match="number of groups"):
            random_partitions(
                range(4), num=1, sizes=[4, 0], group_labels=["a"] * 4, rng=0
            )[0]

    def test_group_sample_sizes_do_not_bias_assignment(self) -> None:
        r"""Either unequal group can be the test set, even if it misses sample targets."""
        labels = ["a"] * 9 + ["b"]
        test_sets = {
            tuple(
                random_partitions(
                    range(10), num=1, ratios=[1, 1], group_labels=labels, rng=seed
                )[0][1]
            )
            for seed in range(16)
        }
        assert test_sets == {tuple(range(9)), (9,)}
        assert (
            random_partitions(
                range(10), num=1, ratios=[9, 1], group_labels=labels, rng=0
            )[0][1]
            == []
        )

    def test_group_error_checks_samples_without_retrying(self) -> None:
        r"""Equal group counts can violate sample targets; failure consumes just one draw."""
        options: dict[str, Any] = {"ratios": [1, 1], "group_labels": ["a"] * 9 + ["b"]}
        rng = Random(37)
        reference_rng = Random(37)
        expected = random_partitions(range(10), num=1, **options, rng=reference_rng)[0]
        assert (
            random_partitions(range(10), num=1, **options, rng=37, max_error=0.41)[0]
            == expected
        )
        with pytest.raises(RuntimeError, match="total variation distance"):
            random_partitions(range(10), num=1, **options, rng=rng, max_error=0.39)[0]
        assert rng.getstate() == reference_rng.getstate()

    @pytest.mark.parametrize("parameter", ["sizes", "ratios"])
    def test_partition_empty_group_labels(self, parameter: str) -> None:
        spec = [0, 0] if parameter == "sizes" else [1, 1]
        options: dict[str, Any] = {parameter: spec}
        if parameter == "sizes":
            with pytest.raises(ValueError, match=r"sizes.*positive"):
                random_partitions([], num=1, **options, group_labels=[], rng=0)
            return
        assert random_partitions([], num=1, **options, group_labels=[], rng=0)[0] == (
            [],
            [],
        )
        assert random_partitions([], num=2, **options, group_labels=[], rng=0) == [
            ([], []),
            ([], []),
        ]

    @pytest.mark.parametrize("sizes", [[-1, 4], [1, 1], [5, 0]])
    def test_random_partitions_invalid_sizes(self, sizes) -> None:
        r"""Repeated draws retain validation of non-negative, exact group counts."""
        with pytest.raises(ValueError, match="sizes"):
            random_partitions(
                range(4), num=2, sizes=sizes, group_labels=[0, 1, 0, 2], rng=0
            )

    def test_repeated_folds_advance_rng(self) -> None:
        r"""A fixed seed reproduces the full sequence without resetting each fold."""
        folds = random_partitions(
            range(8),
            ratios={"train": 2, "valid": 1, "test": 1},
            num=2,
            rng=2022,
        )
        assert folds[0]["train"] == [0, 1, 5, 6]
        assert folds[1]["train"] == [1, 2, 4, 7]
        assert folds[1]["valid"] == [0, 3]
        assert folds[1]["test"] == [5, 6]

    def test_largest_remainder_and_empty_splits(self) -> None:
        folds = random_partitions(
            range(5),
            ratios={"train": 1, "valid": 1, "test": 1, "inference": 0},
            num=2,
            rng=0,
        )
        for fold in range(2):
            assert [
                len(folds[fold][split])
                for split in ("train", "valid", "test", "inference")
            ] == [2, 2, 1, 0]

    def test_extreme_ratios_are_normalized_without_overflow(self) -> None:
        folds = random_partitions(
            range(5),
            ratios={"train": 1e308, "test": 1e308},
            num=1,
            rng=0,
        )
        assert len(folds[0]["train"]) == 3
        assert len(folds[0]["test"]) == 2

    def test_repeated_error_checks_later_folds(self) -> None:
        r"""Fold zero balances items exactly; fold one has TV 0.125 despite equal groups."""
        with pytest.raises(RuntimeError, match=r"total variation distance 0.125"):
            random_partitions(
                range(8),
                ratios=[1, 1],
                group_labels=[0, 0, 0, 1, 2, 2, 3, 3],
                num=2,
                rng=6,
                max_error=0.05,
            )

    def test_random_partition_dictionary_keys(self) -> None:
        r"""Partition dictionary keys into train, validation, and test datasets."""
        dataset = {f"sample-{index}": index for index in range(10)}
        train_keys, validation_keys, test_keys = random_partitions(
            dataset.keys(), num=1, sizes=[4, 4, 2], rng=0
        )[0]

        train_set = {key: dataset[key] for key in train_keys}
        validation_set = {key: dataset[key] for key in validation_keys}
        test_set = {key: dataset[key] for key in test_keys}

        assert tuple(map(len, (train_set, validation_set, test_set))) == (4, 4, 2)
        assert train_set.keys().isdisjoint(validation_set)
        assert train_set.keys().isdisjoint(test_set)
        assert validation_set.keys().isdisjoint(test_set)
        assert (
            train_set.keys() | validation_set.keys() | test_set.keys() == dataset.keys()
        )

    def test_random_partition_tensor_and_labels(self) -> None:
        r"""Partition paired tensor features and string labels by shared indices."""
        features = torch.arange(30).reshape(10, 3)
        labels = [f"label-{index}" for index in range(len(features))]
        train_indices, validation_indices, test_indices = random_partitions(
            range(len(features)), num=1, ratios=[0.4, 0.4, 0.2], rng=0
        )[0]

        train_set = (
            features[train_indices],
            [labels[index] for index in train_indices],
        )
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
            for dataset_features, dataset_labels in (
                train_set,
                validation_set,
                test_set,
            )
            for row, label in zip(dataset_features, dataset_labels, strict=True)
        )

    @pytest.mark.parametrize(
        ("ratios", "message"),
        [
            ({}, "ratios must not all be zero"),
            ({"train": 0}, "ratios must not all be zero"),
            ({"train": -1}, "non-negative"),
            ({"train": float("nan")}, "finite"),
            ({"train": float("inf")}, "finite"),
        ],
    )
    @pytest.mark.parametrize("named", [True, False])
    def test_invalid_ratio_messages(self, ratios, message: str, *, named: bool) -> None:
        with pytest.raises(ValueError, match=message):
            random_partitions(
                [], ratios=ratios if named else list(ratios.values()), num=1, rng=0
            )


class TestRotatatingPartition(TestPartitions):
    r"""Chunk rotation, initial partitions, and coverage across a full cycle."""

    __test__ = True
    function = staticmethod(rotating_partitions)
    parameter_name = "chunks"

    @pytest.mark.parametrize("named", [True, False])
    @pytest.mark.parametrize("grouped", [True, False])
    def test_without_shuffling(self, *, named: bool, grouped: bool) -> None:
        r"""Omitted and explicit None RNGs preserve group order and global RNG state."""
        items = range(8) if grouped else range(4)
        labels = ["a", "b", "a", "c", "b", "d", "c", "d"] if grouped else None
        train, test = ([0, 1, 2, 4], [3, 5, 6, 7]) if grouped else ([0, 1], [2, 3])
        chunks = {"train": 1, "test": 1} if named else [1, 1]
        expected = (
            [{"train": train, "test": test}, {"train": test, "test": train}]
            if named
            else [(train, test), (test, train)]
        )
        initial_state = getstate()
        folds = rotating_partitions(items, num=2, chunks=chunks, group_labels=labels)
        assert folds == expected
        assert (
            rotating_partitions(
                items, num=2, chunks=chunks, rng=None, group_labels=labels
            )
            == expected
        )
        assert getstate() == initial_state

    def test_rotating_fold_assignments(self) -> None:
        r"""Pin seeded assignments and verify coverage across an entire rotation."""
        folds = rotating_partitions(
            range(8),
            chunks={"train": 2, "valid": 1, "test": 1},
            num=4,
            rng=2022,
        )
        assert folds[0]["train"] == [0, 1, 5, 6]
        assert folds[0]["valid"] == [2, 7]
        assert folds[0]["test"] == [3, 4]
        for fold in range(4):
            parts = [folds[fold][split] for split in ("train", "valid", "test")]
            flattened = [key for part in parts for key in part]
            assert sorted(flattened) == list(range(8))
            assert [len(part) for part in parts] == [4, 2, 2]
        for key in range(8):
            assert sum(key in folds[fold]["test"] for fold in range(4)) == 1
            assert sum(key in folds[fold]["valid"] for fold in range(4)) == 1

    @pytest.mark.parametrize(
        ("num", "expected_test"),
        [(4, [3, 4, 1, 0]), (6, [3, 4, 2, 1, 0, 5])],
    )
    @pytest.mark.parametrize("named", [True, False])
    def test_evenly_spaced_rotations(
        self, num: int, expected_test: list[int], *, named: bool
    ) -> None:
        r"""Composite chunk counts allow distinct, evenly spaced fold offsets."""
        chunks = {"train": 5, "test": 1} if named else [5, 1]
        folds = rotating_partitions(range(6), num=num, chunks=chunks, rng=0)
        parts = [
            tuple(fold.values()) if isinstance(fold, dict) else fold for fold in folds
        ]
        assert [test for _, test in parts] == [[item] for item in expected_test]
        for train, test in parts:
            assert len(train) == 5
            assert is_partition((train, test), union=range(6))

    @pytest.mark.parametrize("named", [True, False])
    @pytest.mark.parametrize("grouped", [True, False])
    def test_rotating_single_partition(self, *, named: bool, grouped: bool) -> None:
        r"""An explicit count of one returns a list that supports partition unpacking."""
        marker = object()
        chunks = (
            MappingProxyType({marker: 1, "test": 1, "empty": 0}) if named else [1, 1, 0]
        )
        items = [[index] for index in range(8)]
        labels = [0, 1, 0, 2, 0, 3, 2, 3] if grouped else None
        rng = Random(12)
        reference_rng = Random(12)
        positional = rotating_partitions(
            items, 1, chunks=chunks, group_labels=labels, rng=rng
        )
        singleton = rotating_partitions(
            items, num=1, chunks=chunks, group_labels=labels, rng=reference_rng
        )
        multiple = rotating_partitions(
            items, num=2, chunks=chunks, group_labels=labels, rng=12
        )
        assert isinstance(positional, list)
        assert isinstance(singleton, list)
        assert positional == singleton
        [single] = positional
        assert isinstance(single, dict if named else tuple)
        assert single == multiple[0]
        assert rng.getstate() == reference_rng.getstate()
        if isinstance(single, dict):
            assert list(single) == list(chunks)
            assert next(iter(single)) is marker
            assert single["empty"] == []
        else:
            assert single[-1] == []

    @pytest.mark.parametrize("named", [True, False])
    def test_rotating_single_empty_partition(self, *, named: bool) -> None:
        chunks = {"train": 1, "test": 1} if named else [1, 1]
        expected = {"train": [], "test": []} if named else ([], [])
        assert rotating_partitions([], num=1, chunks=chunks, rng=0) == [expected]

    def test_rotating_single_partition_error_check(self) -> None:
        r"""A single fold retains sample-proportion checks for unequal groups."""
        with pytest.raises(RuntimeError, match="total variation distance"):
            rotating_partitions(
                range(10),
                num=1,
                chunks=[1, 1],
                group_labels=[0] * 9 + [1],
                rng=0,
                max_error=0.05,
            )

    def test_nested_folds_exclude_outer_test_elements(self) -> None:
        outer = rotating_partitions(
            range(12), chunks={"train": 2, "test": 1}, num=3, rng=7
        )
        for fold in range(3):
            inner = rotating_partitions(
                outer[fold]["train"],
                chunks={"train": 1, "valid": 1},
                num=2,
                rng=8,
            )
            assert len(inner) == 2
            for partition in inner:
                for members in partition.values():
                    assert set(members) <= set(outer[fold]["train"])
                    assert set(members).isdisjoint(outer[fold]["test"])

    def test_rotating_partitions_with_boolean_items(self) -> None:
        assert rotating_partitions(
            [True, False], chunks={"train": 1, "test": 1}, num=2, rng=0
        )[0]["train"] == [True]

    @pytest.mark.parametrize(
        ("chunks", "message"),
        [
            ({}, "positive"),
            ({"train": 0}, "positive"),
            ({"train": -1}, "non-negative"),
            ({"train": 1.5}, "integer counts"),
            ({"train": 1}, "must not exceed the total chunk count"),
        ],
    )
    @pytest.mark.parametrize("named", [True, False])
    def test_invalid_chunk_counts(self, chunks, message: str, *, named: bool) -> None:
        with pytest.raises(ValueError, match=message):
            rotating_partitions(
                [], chunks=chunks if named else list(chunks.values()), num=2, rng=0
            )

    def test_rotating_error_checks_later_folds(self) -> None:
        r"""Fold zero has TV 0.05, but fold one has TV 0.15 and must reject the batch."""
        with pytest.raises(RuntimeError, match=r"total variation distance 0.15"):
            rotating_partitions(range(10), chunks=[6, 2], num=4, rng=0, max_error=0.125)
