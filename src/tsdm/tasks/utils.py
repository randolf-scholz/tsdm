r"""Reproducible partitions of collection items.

Items need not be unique or hashable. Optional hashable group labels, aligned
with items by iteration order, keep several occurrences in the same split.
Split proportions apply to groups when supplied, otherwise to individual occurrences.
Optional `max_error` bounds the total variation distance of item proportions from
the target weights. Checking rejects empty inputs; explicit `sizes` bypass it.

Both fold generators assign every occurrence exactly once per fold, retain empty
splits, and preserve input order within each split. Reproducibility requires
the same seed or initial RNG state, ordered items, split configuration,
group labels, and Python random implementation. Unordered sets of items or labels
are rejected by the fold generators. RNGs constructed from integer seeds do not
affect global random state. These stdlib algorithms do not reproduce historical
NumPy assignments; persist assignments when exact reproduction across algorithm
or runtime changes is required.

Mappings specify named splits; results preserve their exact keys and iteration
order. Sequences specify unnamed splits; results preserve their length and order.
Partitions are dictionaries for named splits and tuples for unnamed splits. Fold
generators return a list of partitions, one per fold. For nested cross-validation,
pass only the outer training items to the inner generator to prevent leakage.
"""

__all__ = [
    "repeated_random_folds",
    "rotating_folds",
    "random_partition",
    "is_partition",
]

from collections.abc import Collection, Iterable, Mapping, Sequence, Set as AbstractSet
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
    items: Collection, num_folds: int, /, *, groups: Collection[Label] | None
) -> None:
    r"""Validate fold counts and the ordered, positionally aligned inputs."""
    if not isinstance(num_folds, int) or num_folds <= 0:
        raise ValueError("num_folds must be a positive integer.")
    if isinstance(items, AbstractSet):
        raise TypeError("items must have a reproducible iteration order, not a set.")
    if isinstance(groups, AbstractSet):
        raise TypeError("groups must have a reproducible iteration order, not a set.")
    if groups is not None and len(groups) != len(items):
        raise ValueError("groups must have the same length as items.")


@overload
def rotating_folds[T, K](
    items: Collection[T],
    /,
    chunks: Mapping[K, int],
    *,
    num_folds: int,
    rng: int | Random,
    groups: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[dict[K, list[T]]]: ...
@overload
def rotating_folds[T](
    items: Collection[T],
    /,
    chunks: Sequence[int],
    *,
    num_folds: int,
    rng: int | Random,
    groups: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[tuple[list[T], ...]]: ...
def rotating_folds[T, K](
    items: Collection[T],
    /,
    chunks: Mapping[K, int] | Sequence[int],
    *,
    num_folds: int,
    rng: int | Random,
    groups: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[dict[K, list[T]]] | list[tuple[list[T], ...]]:
    r"""Partition collection elements by rotating shuffled group chunks.

    Shuffle distinct groups once and distribute them over the configured total
    number of chunks. Chunk sizes differ by at most one; earlier chunks receive leftovers.
    Each fold shifts split boundaries by `num_chunks // num_folds` chunks.
    This preserves the rotation policy of `datatools.folds_from_groups`.

    Args:
        items: Ordered collection of items; duplicates and unhashable items are
            allowed. Without `groups`, every occurrence is split independently.
        chunks: Mapping from split keys to non-negative integer chunk counts, or
            a sequence of counts. Their positive sum must be divisible by
            `num_folds`. Mapping keys are preserved exactly without normalization.
        num_folds: Positive number of folds.
        rng: Required integer seed or existing `random.Random` instance. An
            integer creates a local RNG; an existing instance advances in place.
        groups: Optional collection of stable, hashable labels, one per item in
            iteration order. Must have the same length as `items`. Occurrences
            with equal labels stay together. Labels need not support ordering.
        max_error: Maximum item-proportion TV distance from `chunks`, in `[0, 1]`.
            `None` disables checking; exceeding the bound raises `RuntimeError`.

    Returns:
        A list of `num_folds` partitions. Each partition is a dictionary with the
        same keys and order as `chunks` for a mapping, or a tuple with the same
        length and order for a sequence. Each split contains a list of items.

    Splits assigned zero chunks are retained as empty lists. Too few
    groups can produce empty splits, including training splits. Group counts,
    rather than element counts, determine the proportions when grouping is used.
    A one-chunk test split with `num_folds` chunks visits every group once;
    arbitrary chunk counts do not guarantee equal coverage across folds.

    For hashable items, pass `groups=items` to keep equal values together. For
    unhashable items, supply hashable labels. For lists of hashable values, use
    `groups=[tuple(item) for item in items]`. Equal items can occur in different splits
    unless their labels agree. Grouping preserves every occurrence.

    Examples:
        Rotate training and test splits over two folds for four elements.

        ```pycon
        >>> from pprint import pprint
        >>> folds = rotating_folds(
        ...     range(4),
        ...     chunks={"train": 1, "test": 1},
        ...     num_folds=2,
        ...     rng=2022,
        ... )
        >>> pprint(folds, sort_dicts=False)
        [{'train': [0, 3], 'test': [1, 2]}, {'train': [1, 2], 'test': [0, 3]}]

        >>> rotating_folds(range(4), chunks=[1, 1], num_folds=2, rng=2022)
        [([0, 3], [1, 2]), ([1, 2], [0, 3])]

        ```
    """
    _validate_args(items, num_folds, groups=groups)
    rng = _normalize_rng(rng)
    counts = list(chunks.values() if isinstance(chunks, Mapping) else chunks)
    _validate_weights(counts)
    num_chunks = sum(counts)

    if not all(isinstance(count, int) for count in counts):
        raise ValueError("chunks must contain non-negative integer counts.")
    if num_chunks % num_folds:
        raise ValueError("Total chunk count must be divisible by num_folds.")

    # Deduplicate groups in element order, then shuffle once for all folds.
    labels = list(range(len(items))) if groups is None else list(groups)
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
    step = num_chunks // num_folds
    result: list[tuple[list[T], ...]] = []

    for fold in range(num_folds):
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


@overload
def repeated_random_folds[T, K](
    items: Collection[T],
    /,
    ratios: Mapping[K, float],
    *,
    num_folds: int,
    rng: int | Random,
    groups: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[dict[K, list[T]]]: ...
@overload
def repeated_random_folds[T](
    items: Collection[T],
    /,
    ratios: Sequence[float],
    *,
    num_folds: int,
    rng: int | Random,
    groups: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[tuple[list[T], ...]]: ...
def repeated_random_folds[T, K](
    items: Collection[T],
    /,
    ratios: Mapping[K, float] | Sequence[float],
    *,
    num_folds: int,
    rng: int | Random,
    groups: Collection[Label] | None = None,
    max_error: Optional[float] = None,
) -> list[dict[K, list[T]]] | list[tuple[list[T], ...]]:
    r"""Create repeated random partitions of collection elements.

    Args:
        items: Ordered collection of items; duplicates and unhashable items are
            allowed. Without `groups`, every occurrence is split independently.
        ratios: Mapping from split keys to finite, non-negative relative weights,
            or a sequence of weights, with at least one positive weight. Mapping
            keys are preserved exactly without normalization. Allocate group
            counts by largest remainder, breaking ties by split order.
        num_folds: Positive number of partitions to draw.
        rng: Required integer seed or existing `random.Random` instance. One RNG
            advances across folds; an existing instance also retains its state
            between calls. Partitions may coincide by chance for small inputs.
        groups: Optional collection of stable, hashable labels, one per item in
            iteration order. Must have the same length as `items`. Occurrences
            with equal labels stay together. Labels need not support ordering.
        max_error: Maximum item-proportion TV distance from `ratios`, in `[0, 1]`.
            `None` disables checking; exceeding the bound raises `RuntimeError`.

    Returns:
        A list of `num_folds` partitions. Each partition is a dictionary with the
        same keys and order as `ratios` for a mapping, or a tuple with the same
        length and order for a sequence. Each split contains a list of items.

    These are repeated holdouts: test sets can overlap across folds. No rotation
    coverage is guaranteed. Zero-ratio splits yield empty lists.
    Group proportions need not equal collection-element proportions.

    For hashable items, pass `groups=items` to keep equal values together. For
    unhashable items, supply hashable labels. For lists of hashable values, use
    `groups=[tuple(item) for item in items]`. Equal items can occur in different splits
    unless their labels agree. Grouping preserves every occurrence.

    Examples:
        Draw two equal-proportion training and test partitions of four elements.

        ```pycon
        >>> from pprint import pprint
        >>> folds = repeated_random_folds(
        ...     range(4),
        ...     ratios={"train": 1, "test": 1},
        ...     num_folds=2,
        ...     rng=2022,
        ... )
        >>> pprint(folds, sort_dicts=False)
        [{'train': [0, 3], 'test': [1, 2]}, {'train': [1, 3], 'test': [0, 2]}]

        >>> repeated_random_folds(range(4), ratios=[1, 1], num_folds=2, rng=2022)
        [([0, 3], [1, 2]), ([1, 3], [0, 2])]

        ```
    """
    _validate_args(items, num_folds, groups=groups)
    rng = _normalize_rng(rng)
    weights = list(ratios.values() if isinstance(ratios, Mapping) else ratios)
    _validate_weights(weights)
    scale = max(weights)

    # Count distinct groups in element order. Without grouping, each element
    # contributes one group, so the same allocation procedure applies.
    labels = list(range(len(items))) if groups is None else list(groups)
    unique_groups = list(dict.fromkeys(labels))

    # Scale ratios before summing to avoid overflow for large finite ratios.
    normalized = [ratio / scale for ratio in weights]
    total = fsum(normalized)
    quotas = [len(unique_groups) * ratio / total for ratio in normalized]
    counts = [floor(quota) for quota in quotas]

    # Allocate leftover groups to the largest fractional quotas. Python's stable
    # sort breaks ties by the insertion order of the configured splits.
    remainder_order = sorted(
        range(len(weights)), key=lambda i: quotas[i] - counts[i], reverse=True
    )
    for index in remainder_order[: len(unique_groups) - sum(counts)]:
        counts[index] += 1

    # Advance one local RNG across folds instead of reseeding it for each fold.
    result: list[tuple[list[T], ...]] = []

    for _ in range(num_folds):
        # Shuffle a fresh copy so each draw starts from the same group order.
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)

        # Assign consecutive groups to splits using the precomputed counts.
        assignments: dict[Label, int] = {}
        start = 0
        for split, count in enumerate(counts):
            stop = start + count
            assignments.update(dict.fromkeys(shuffled[start:stop], split))
            start = stop

        # Retain empty splits, then collect items in one pass to preserve
        # input order without scanning all elements separately for each split.
        parts: tuple[list[T], ...] = tuple([] for _ in counts)
        for item, label in zip(items, labels, strict=True):
            split = assignments[label]
            parts[split].append(item)
        if max_error is not None:
            _check_partition_error(
                [len(p) for p in parts], weights, max_error=max_error
            )
        result.append(parts)

    if isinstance(ratios, Mapping):
        return [dict(zip(ratios, parts, strict=True)) for parts in result]
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
    max_error: Optional[float] = None,
) -> dict[K, list[T]]: ...
@overload
def random_partition[T, K](
    items: Collection[T],
    *,
    ratios: Mapping[K, SupportsFloat],
    rng: int | Random,
    max_error: Optional[float] = None,
) -> dict[K, list[T]]: ...
@overload
def random_partition[T](
    items: Collection[T],
    *,
    sizes: Sequence[SupportsIndex],
    rng: int | Random,
    max_error: Optional[float] = None,
) -> tuple[list[T], ...]: ...
@overload
def random_partition[T](
    items: Collection[T],
    *,
    ratios: Sequence[SupportsFloat],
    rng: int | Random,
    max_error: Optional[float] = None,
) -> tuple[list[T], ...]: ...
def random_partition[T, K](
    items: Collection[T],
    *,
    sizes: Mapping[K, SupportsIndex] | Sequence[SupportsIndex] | None = None,
    ratios: Mapping[K, SupportsFloat] | Sequence[SupportsFloat] | None = None,
    rng: int | Random,
    max_error: Optional[float] = None,
) -> dict[K, list[T]] | tuple[list[T], ...]:
    r"""Create a random partition of item occurrences.

    Exactly one of `sizes` or `ratios` must be provided. Sizes are absolute
    partition counts and must sum to the number of items. Ratios are normalized
    weights used to distribute all occurrences between the partitions.

    Args:
        items: Items to distribute among the partitions. Duplicates and unhashable
            items are allowed; each occurrence is split independently.
        sizes: Mapping from split keys to absolute counts, or a sequence of
            counts. The counts must be non-negative and
            sum to the number of items. Mutually exclusive with `ratios`.
        ratios: Mapping from split keys to relative weights, or a sequence of
            weights. Weights must be finite, non-negative and not all zero.
            Mutually exclusive with `sizes`. Largest remainder allocates counts,
            breaking ties in favor of later splits.
        rng: Required integer seed or existing `random.Random` instance. An
            integer creates a local RNG; an existing instance advances in place.
        max_error: Maximum item-proportion TV distance from `ratios`, in `[0, 1]`.
            `None` disables checking; exceeding the bound raises `RuntimeError`.
            Ignored with explicit `sizes`.

    Returns:
        A dictionary with the exact keys and order of the supplied mapping, or a
        tuple with the same length and order as the supplied sequence. Each split
        contains a list of items in shuffled order. Empty splits are retained.

    Examples:
        ```pycon
        >>> random_partition(range(4), sizes=[2, 2], rng=0)
        ([2, 0], [1, 3])
        >>> random_partition(range(4), ratios={"train": 1, "test": 1}, rng=0)
        {'train': [2, 0], 'test': [1, 3]}

        ```
    """
    rng = _normalize_rng(rng)
    num = len(items)

    match sizes, ratios:
        case [sizes, None] if sizes is not None:
            counts = [
                size.__index__()
                for size in (sizes.values() if isinstance(sizes, Mapping) else sizes)
            ]
            if any(count < 0 for count in counts) or sum(counts) != num:
                raise ValueError(
                    "sizes must be non-negative and sum to the number of items"
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

            # Give later partitions precedence when fractional remainders tie.
            if remainder := num - sum(counts):
                order = sorted(range(len(counts)), key=lambda i: quotas[i] - counts[i])
                for index in order[-remainder:]:
                    counts[index] += 1
            if max_error is not None:
                _check_partition_error(counts, weights, max_error=max_error)
        case _:
            raise ValueError("Exactly one of sizes or ratios must be provided")

    shuffled = list(items)
    rng.shuffle(shuffled)
    partitions = []
    start = 0
    for count in counts:
        stop = start + count
        partitions.append(shuffled[start:stop])
        start = stop
    if isinstance(sizes, Mapping):
        return dict(zip(sizes, partitions, strict=True))
    if isinstance(ratios, Mapping):
        return dict(zip(ratios, partitions, strict=True))
    return tuple(partitions)
