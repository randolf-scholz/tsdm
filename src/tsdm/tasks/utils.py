r"""Reproducible partitions of collection items.

Items need not be unique or hashable. Optional hashable group labels, aligned
with items by iteration order, keep several occurrences in the same split.
Split proportions apply to groups when supplied, otherwise to individual occurrences.
Optional `max_error` bounds the total variation distance of item proportions from
the target weights. Checking rejects empty inputs; explicit `sizes` bypass it.

Group labels can mark redundant samples, such as results from replication
experiments. Samples with the same label always belong to the same split to
avoid data leakage. Labels describe groups to keep together, not classes for
stratification. All three partition generators support `group_labels`.

Both fold generators assign every occurrence exactly once per fold, retain empty
splits, and preserve input order within each split. Reproducibility requires
the same seed or initial RNG state, ordered items, split configuration,
group labels, and Python random implementation. Unordered sets of items or labels
are rejected by the partition generators. RNGs constructed from integer seeds do not
affect global random state. These stdlib algorithms do not reproduce historical
NumPy assignments; persist assignments when exact reproduction across algorithm
or runtime changes is required.

Mappings specify named splits; results preserve their exact keys and iteration
order. Sequences specify unnamed splits; results preserve their length and order.
Partitions are dictionaries for named splits and tuples for unnamed splits.
Both fold generators return a list of partitions, one per fold.
`rotating_folds` defaults to one fold. For nested cross-validation,
pass only the outer training items to the inner generator to prevent leakage.
"""

__all__ = [
    "random_partitions",
    "rotating_folds",
    "random_partition",
    "is_partition",
]

from collections.abc import (
    Collection,
    Iterable,
    KeysView,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from math import floor, fsum, isfinite
from random import Random
from typing import Optional, SupportsFloat, SupportsIndex, overload

from tsdm.stats import total_variation_distance

type Label = object


def _normalize_rng(rng: int | Random, /) -> Random:
    r"""Create a local RNG from a seed or reuse an existing instance unchanged."""
    match rng:
        case Random():
            return rng
        case int():
            return Random(rng)
        case _:
            raise TypeError("rng must be an integer seed or a random.Random instance.")


def _validate_weights(values: Sequence[float], /) -> None:
    r"""Require finite, non-negative weights with a positive total."""
    if not values:
        raise ValueError("At least one split must be configured.")
    if any(not isfinite(weight) for weight in values):
        raise ValueError("Split weights must be finite.")
    if any(weight < 0 for weight in values):
        raise ValueError("Split weights must be non-negative.")
    if not any(weight > 0 for weight in values):
        raise ValueError("At least one split must have a positive weight.")


def _check_partition_error(
    p: Sequence[float], q: Sequence[float], /, *, max_error: float
) -> None:
    r"""Check normalized item counts against the target weights."""
    if not (0 <= max_error <= 1):
        raise ValueError("max_error must be between 0 and 1.")

    error = total_variation_distance(p, q).item()
    if error > max_error:
        raise RuntimeError(
            f"total variation distance {error:.6g} exceeds max_error={max_error:.6g}."
        )


def _validate_args(
    items: Collection, num: int, /, *, group_labels: Collection[Label] | None
) -> None:
    r"""Validate fold counts and the ordered, positionally aligned inputs."""
    if not isinstance(num, int) or num <= 0:
        raise ValueError("num must be a positive integer.")
    _validate_inputs(items, group_labels)


def _validate_inputs(
    items: Collection, group_labels: Collection[Label] | None, /
) -> None:
    r"""Require ordered items and positionally aligned group labels."""
    if isinstance(items, AbstractSet) and not isinstance(items, KeysView):
        raise TypeError("items must have a reproducible iteration order, not a set.")
    if isinstance(group_labels, AbstractSet) and not isinstance(group_labels, KeysView):
        raise TypeError(
            "group_labels must have a reproducible iteration order, not a set."
        )
    if group_labels is not None and len(group_labels) != len(items):
        raise ValueError("group_labels must have the same length as items.")


@overload
def rotating_folds[T, K](
    items: Collection[T],
    /,
    num: int = 1,
    *,
    chunks: Mapping[K, int],
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[dict[K, list[T]]]: ...
@overload
def rotating_folds[T](
    items: Collection[T],
    /,
    num: int = 1,
    *,
    chunks: Sequence[int],
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[tuple[list[T], ...]]: ...
def rotating_folds[T, K](
    items: Collection[T],
    /,
    num: int = 1,
    *,
    chunks: Mapping[K, int] | Sequence[int],
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[dict[K, list[T]]] | list[tuple[list[T], ...]]:
    r"""Partition collection elements by rotating shuffled group chunks.

    Shuffle distinct groups once and distribute them over the configured total
    number of chunks. Chunk sizes differ by at most one; earlier chunks receive leftovers.
    Each fold shifts split boundaries by `num_chunks // num` chunks.
    This preserves the rotation policy of `datatools.folds_from_groups`.
    By default, return a one-element list containing the initial partition.

    Splits assigned zero chunks are retained as empty lists. Too few
    groups can produce empty splits, including training splits. Group counts,
    rather than element counts, determine the proportions when grouping is used.
    A one-chunk test split with `num` chunks visits every group once;
    arbitrary chunk counts do not guarantee equal coverage across folds.

    Group labels can mark redundant samples, such as results from replication
    experiments. Samples with the same label always share a split to avoid data
    leakage.

    For hashable items, pass `group_labels=items` to keep equal values together. For
    unhashable items, supply hashable labels. For lists of hashable values, use
    `group_labels=[tuple(item) for item in items]`. Equal items can occur in different splits
    unless their labels agree. Grouping preserves every occurrence.

    Args:
        items: Ordered collection of items; duplicates and unhashable items are
            allowed. Without `group_labels`, every occurrence is split independently.
        num: Positive number of folds. Defaults to 1.
        chunks: Mapping from split keys to non-negative integer chunk counts, or
            a sequence of counts. Their positive sum must be divisible by
            `num`. Mapping keys are preserved exactly without normalization.
        rng: Required integer seed or existing `random.Random` instance. An
            integer creates a local RNG; an existing instance advances in place.
        group_labels: Optional collection of stable, hashable labels, one per item in
            iteration order. Must have the same length as `items`. Occurrences
            with equal labels stay together. Labels need not support ordering.
        max_error: Maximum item-proportion TV distance from `chunks`, in `[0, 1]`.
            `None` disables checking; exceeding the bound raises `RuntimeError`.

    Returns:
        A list of `num` partitions, including when `num=1`. Each partition is a
        dictionary with the same keys and order as `chunks` for a mapping, or a
        tuple with the same length and order for a sequence. Each split contains
        a list of items.

    Examples:
        Rotate training and test splits over two folds for four elements.

        ```pycon
        >>> from pprint import pprint
        >>> folds = rotating_folds(
        ...     range(4),
        ...     chunks={"train": 1, "test": 1},
        ...     num=2,
        ...     rng=2022,
        ... )
        >>> pprint(folds, sort_dicts=False)
        [{'train': [0, 3], 'test': [1, 2]}, {'train': [1, 2], 'test': [0, 3]}]

        >>> rotating_folds(range(4), chunks=[1, 1], num=2, rng=2022)
        [([0, 3], [1, 2]), ([1, 2], [0, 3])]

        >>> rotating_folds(range(4), chunks=[1, 1], rng=2022)
        [([0, 3], [1, 2])]
        >>> [partition] = rotating_folds(range(4), chunks=[1, 1], rng=2022)
        >>> partition
        ([0, 3], [1, 2])

        ```
    """
    _validate_args(items, num, group_labels=group_labels)
    rng = _normalize_rng(rng)
    counts = list(chunks.values() if isinstance(chunks, Mapping) else chunks)
    _validate_weights(counts)
    num_chunks = sum(counts)

    if not all(isinstance(count, int) for count in counts):
        raise ValueError("chunks must contain non-negative integer counts.")
    if num_chunks % num:
        raise ValueError("Total chunk count must be divisible by num.")

    # Deduplicate groups in element order, then shuffle once for all folds.
    labels = list(range(len(items))) if group_labels is None else list(group_labels)
    unique_groups = list(dict.fromkeys(labels))
    rng.shuffle(unique_groups)

    # Divide the shuffled groups into chunks. The first few chunks receive one
    # extra group when the group count is not divisible by the chunk count.
    base_size, num_larger_chunks = divmod(len(unique_groups), num_chunks)
    chunk_by_group: dict[Label, int] = {}
    start = 0

    for chunk in range(num_chunks):
        chunk_size = base_size + (1 if chunk < num_larger_chunks else 0)
        stop = start + chunk_size
        chunk_by_group.update(dict.fromkeys(unique_groups[start:stop], chunk))
        start = stop

    # Each split reserves its configured number of consecutive chunks in fold zero.
    # Later folds move these boundaries by a fixed number of chunks.
    chunk_splits = [split for split, count in enumerate(counts) for _ in range(count)]
    step = num_chunks // num
    result: list[tuple[list[T], ...]] = []

    for fold in range(num):
        # Retain empty splits, then rotate chunk boundaries and collect
        # items in their original order in one pass.
        parts: tuple[list[T], ...] = tuple([] for _ in counts)
        for item, label in zip(items, labels, strict=True):
            chunk = chunk_by_group[label]
            split = chunk_splits[(chunk - fold * step) % num_chunks]
            parts[split].append(item)
        if max_error is not None:
            _check_partition_error([len(p) for p in parts], counts, max_error=max_error)
        result.append(parts)

    if isinstance(chunks, Mapping):
        return [dict(zip(chunks, parts, strict=True)) for parts in result]
    return result


def is_partition[T](
    partition: Iterable[Collection[T]],
    /,
    *,
    union: Collection[T],
) -> bool:
    r"""Check that the parts are disjoint and contain no duplicate elements.

    If `union` is provided, also check that the parts cover it exactly.
    The input must support repeated iteration because it is traversed twice.
    """
    sets = (set(p) for p in partition)
    part_union = set().union(*sets)

    if part_union != set(union):
        return False
    return len(part_union) == sum(len(p) for p in partition)


@overload
def random_partition[T, K](
    items: Collection[T],
    *,
    sizes: Mapping[K, SupportsIndex],
    rng: int | Random,
    group_labels: Optional[Collection[Label]] = None,
    max_error: Optional[float] = None,
) -> dict[K, list[T]]: ...
@overload
def random_partition[T, K](
    items: Collection[T],
    *,
    ratios: Mapping[K, SupportsFloat],
    rng: int | Random,
    group_labels: Optional[Collection[Label]] = None,
    max_error: Optional[float] = None,
) -> dict[K, list[T]]: ...
@overload
def random_partition[T](
    items: Collection[T],
    *,
    sizes: Sequence[SupportsIndex],
    rng: int | Random,
    group_labels: Optional[Collection[Label]] = None,
    max_error: Optional[float] = None,
) -> tuple[list[T], ...]: ...
@overload
def random_partition[T](
    items: Collection[T],
    *,
    ratios: Sequence[SupportsFloat],
    rng: int | Random,
    group_labels: Optional[Collection[Label]] = None,
    max_error: Optional[float] = None,
) -> tuple[list[T], ...]: ...
def random_partition[T, K](
    items: Collection[T],
    *,
    sizes: Mapping[K, SupportsIndex] | Sequence[SupportsIndex] | None = None,
    ratios: Mapping[K, SupportsFloat] | Sequence[SupportsFloat] | None = None,
    rng: int | Random,
    group_labels: Optional[Collection[Label]] = None,
    max_error: Optional[float] = None,
) -> dict[K, list[T]] | tuple[list[T], ...]:
    r"""Create a random partition of indivisible groups of items.

    Group labels can mark redundant samples, such as results from replication
    experiments. Samples with the same label always share a split to avoid data
    leakage. Without labels, each item occurrence forms its own group.

    Exactly one of `sizes` or `ratios` must be provided. Both refer to group
    counts. For ratios, compute each quota as the number of groups times its
    normalized weight, floor the quotas, and allocate leftover groups by largest
    fractional remainder. Ties favor earlier splits. Shuffle distinct groups in
    first-occurrence order and assign consecutive groups using these counts.
    All assignments with the configured group counts are equally likely.
    Collect items in their original order within each split.

    Group sizes do not influence allocation. Sample proportions can differ
    substantially from the targets; positive ratios can yield empty splits.
    For example, two groups with train/test ratios of 9:1 both go to training.
    `max_error` checks sample proportions and raises on failure without retrying
    or optimizing the assignment.

    Args:
        items: Ordered collection of items to distribute among the partitions.
            Duplicates and unhashable items are allowed. Sets are rejected.
        sizes: Mapping from split keys to absolute group counts, or a sequence
            of counts. Counts must be non-negative and sum to the number of
            distinct groups (the number of items without labels).
            Mutually exclusive with `ratios`.
        ratios: Mapping from split keys to relative weights, or a sequence of
            weights. Weights must be finite, non-negative and not all zero.
            Mutually exclusive with `sizes`. Largest remainder allocates counts,
            breaking ties in favor of earlier splits.
        rng: Required integer seed or existing `random.Random` instance. An
            integer creates a local RNG; an existing instance advances in place.
        group_labels: Optional ordered collection of stable, hashable labels,
            one per item in iteration order. Must have the same length as `items`.
            Equal labels keep occurrences together. Labels need not be sortable.
        max_error: Maximum item-proportion TV distance from `ratios`, in `[0, 1]`.
            `None` disables checking; exceeding the bound raises `RuntimeError`.
            Ignored with explicit `sizes`.

    Returns:
        A dictionary with the exact keys and order of the supplied mapping, or a
        tuple with the same length and order as the supplied sequence. Each split
        contains a list of items in input order. Empty splits and every item
        occurrence are retained.

    Examples:
        ```pycon
        >>> random_partition(range(4), sizes=[2, 2], rng=0)
        ([0, 2], [1, 3])
        >>> random_partition(range(4), ratios={"train": 1, "test": 1}, rng=0)
        {'train': [0, 2], 'test': [1, 3]}
        >>> # Items with the same group label count as one unit: "a1" and "a2" form one group.
        >>> random_partition(
        ...     ["a1", "b1", "a2", "c1"],
        ...     sizes=[1, 2],
        ...     group_labels=["a", "b", "a", "c"],
        ...     rng=0,
        ... )
        (['a1', 'a2'], ['b1', 'c1'])

        ```
    """
    _validate_inputs(items, group_labels)
    rng = _normalize_rng(rng)
    labels = list(range(len(items))) if group_labels is None else list(group_labels)
    unique_groups = list(dict.fromkeys(labels))
    num = len(unique_groups)
    weights: list[float] = []

    match sizes, ratios:
        case [sizes, None] if sizes is not None:
            counts = [
                size.__index__()
                for size in (sizes.values() if isinstance(sizes, Mapping) else sizes)
            ]
            if any(count < 0 for count in counts) or sum(counts) != num:
                raise ValueError(
                    "sizes must be non-negative and sum to the number of groups"
                )
        case [None, ratios] if ratios is not None:
            weights = [
                ratio.__float__()
                for ratio in (
                    ratios.values() if isinstance(ratios, Mapping) else ratios
                )
            ]

            if any(not isfinite(weight) or weight < 0 for weight in weights):
                raise ValueError("ratios must be finite and non-negative")
            scale = max(weights, default=0.0)
            if scale <= 0:
                raise ValueError("ratios must not all be zero")

            # Scale before summing to avoid overflow for large finite weights.
            normalized = [weight / scale for weight in weights]
            total = fsum(normalized)
            quotas = [num * weight / total for weight in normalized]
            counts = [floor(quota) for quota in quotas]

            # Stable descending order gives earlier splits precedence on ties.
            if remainder := num - sum(counts):
                order = sorted(
                    range(len(counts)),
                    key=lambda i: quotas[i] - counts[i],
                    reverse=True,
                )
                for index in order[:remainder]:
                    counts[index] += 1
        case _:
            raise ValueError("Exactly one of sizes or ratios must be provided")

    rng.shuffle(unique_groups)
    assignments: dict[Label, int] = {}
    start = 0
    for split, count in enumerate(counts):
        stop = start + count
        assignments.update(dict.fromkeys(unique_groups[start:stop], split))
        start = stop

    # Membership is random; collecting in one pass retains input order per split.
    partitions: list[list[T]] = [[] for _ in counts]
    for item, label in zip(items, labels, strict=True):
        partitions[assignments[label]].append(item)
    if ratios is not None and max_error is not None:
        _check_partition_error(
            [len(part) for part in partitions], weights, max_error=max_error
        )
    if isinstance(sizes, Mapping):
        return dict(zip(sizes, partitions, strict=True))
    if isinstance(ratios, Mapping):
        return dict(zip(ratios, partitions, strict=True))
    return tuple(partitions)


@overload
def random_partitions[T, K](
    items: Collection[T],
    /,
    num: int,
    *,
    ratios: Mapping[K, float],
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[dict[K, list[T]]]: ...
@overload
def random_partitions[T](
    items: Collection[T],
    /,
    num: int,
    *,
    ratios: Sequence[float],
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[tuple[list[T], ...]]: ...
def random_partitions[T, K](
    items: Collection[T],
    /,
    num: int,
    *,
    ratios: Mapping[K, float] | Sequence[float],
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[dict[K, list[T]]] | list[tuple[list[T], ...]]:
    r"""Create repeated random partitions of collection elements.

    These are repeated holdouts: test sets can overlap across folds. No rotation
    coverage is guaranteed. Zero-ratio splits yield empty lists.
    Group proportions need not equal collection-element proportions.
    Each fold calls `random_partition` using the same advancing RNG.

    Group labels can mark redundant samples, such as results from replication
    experiments. Samples with the same label always share a split to avoid data
    leakage.

    Allocation first counts distinct groups in first-occurrence order. For G
    groups and normalized split weights pᵢ, compute quotas G pᵢ and floor them.
    Allocate the remaining groups to the largest fractional remainders, breaking
    ties in favor of earlier splits. Each fold independently shuffles the groups
    and assigns consecutive groups according to these fixed counts. Conditional
    on the counts, all assignments are equally likely. Group sizes do not affect
    allocation, so sample proportions can differ substantially from the targets.

    Positive weights do not guarantee nonempty splits: with just two groups and
    train/test weights of 9:1, quotas of 1.8 and 0.2 yield two training groups and
    an empty test split in every fold. Even when several assignments are possible,
    repeated partitions may coincide by chance. `max_error` checks the resulting
    sample proportions and raises on failure; it does not retry or optimize the
    assignment to meet sample-size targets.

    For hashable items, pass `group_labels=items` to keep equal values together. For
    unhashable items, supply hashable labels. For lists of hashable values, use
    `group_labels=[tuple(item) for item in items]`. Equal items can occur in different splits
    unless their labels agree. Grouping preserves every occurrence.

    Args:
        items: Ordered collection of items; duplicates and unhashable items are
            allowed. Without `group_labels`, every occurrence is split independently.
        num: Positive number of partitions to draw.
        ratios: Mapping from split keys to finite, non-negative relative weights,
            or a sequence of weights, with at least one positive weight. Mapping
            keys are preserved exactly without normalization. Allocate group
            counts by largest remainder, breaking ties by split order.
        rng: Required integer seed or existing `random.Random` instance. One RNG
            advances across folds; an existing instance also retains its state
            between calls. Partitions may coincide by chance for small inputs.
        group_labels: Optional collection of stable, hashable labels, one per item in
            iteration order. Must have the same length as `items`. Occurrences
            with equal labels stay together. Labels need not support ordering.
        max_error: Maximum item-proportion TV distance from `ratios`, in `[0, 1]`.
            `None` disables checking; exceeding the bound raises `RuntimeError`.

    Returns:
        A list of `num` partitions. Each partition is a dictionary with the
        same keys and order as `ratios` for a mapping, or a tuple with the same
        length and order for a sequence. Each split contains a list of items.

    Examples:
        Draw two equal-proportion training and test partitions of four elements.

        ```pycon
        >>> from pprint import pprint
        >>> folds = random_partitions(
        ...     range(4),
        ...     ratios={"train": 1, "test": 1},
        ...     num=2,
        ...     rng=2022,
        ... )
        >>> pprint(folds, sort_dicts=False)
        [{'train': [0, 3], 'test': [1, 2]}, {'train': [1, 3], 'test': [0, 2]}]

        >>> random_partitions(range(4), ratios=[1, 1], num=2, rng=2022)
        [([0, 3], [1, 2]), ([1, 3], [0, 2])]

        ```
    """
    if not isinstance(num, int) or num <= 0:
        raise ValueError("num must be a positive integer.")
    rng = _normalize_rng(rng)
    return [
        random_partition(
            items,
            ratios=ratios,
            rng=rng,
            group_labels=group_labels,
            max_error=max_error,
        )
        for _ in range(num)
    ]
