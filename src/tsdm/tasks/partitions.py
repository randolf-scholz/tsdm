r"""Reproducible partitions of collection items.

Items need not be unique or hashable. Optional hashable group labels, aligned
with items by iteration order, keep several occurrences in the same split.
Split proportions apply to groups when supplied, otherwise to individual occurrences.
Optional `max_error` bounds the total variation distance of item proportions from
the target weights. Checking rejects empty inputs; explicit `sizes` bypass it.

Group labels can mark redundant samples, such as results from replication
experiments. Samples with the same label always belong to the same split to
avoid data leakage. Labels describe groups to keep together, not classes for
stratification. Both partition generators support `group_labels`.

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
Both require an explicit positive `num`. For nested cross-validation,
pass only the outer training items to the inner generator to prevent leakage.
"""

__all__ = [
    "random_partitions",
    "rotating_partitions",
    "is_partition",
]

import operator
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
from typing import SupportsFloat, SupportsIndex, overload

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


def _normalize_counts[K](
    values: Mapping[K, SupportsIndex] | Sequence[SupportsIndex], /, *, name: str
) -> list[int]:
    r"""Convert index-compatible counts to non-negative integers with a positive total."""
    try:
        counts = [
            operator.index(value)  # avoids AttributeError compared to value.__index__()
            for value in (values.values() if isinstance(values, Mapping) else values)
        ]
    except TypeError as exc:
        raise ValueError(f"{name} must contain non-negative integer counts.") from exc
    if any(count < 0 for count in counts):
        raise ValueError(f"{name} must contain non-negative integer counts.")
    if not any(count > 0 for count in counts):
        raise ValueError(f"{name} must contain at least one positive count.")
    return counts


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
    if isinstance(items, AbstractSet) and not isinstance(items, KeysView):
        raise TypeError("items must have a reproducible iteration order, not a set.")
    if isinstance(group_labels, AbstractSet) and not isinstance(group_labels, KeysView):
        raise TypeError(
            "group_labels must have a reproducible iteration order, not a set."
        )
    if group_labels is not None and len(group_labels) != len(items):
        raise ValueError("group_labels must have the same length as items.")


@overload
def rotating_partitions[T, K](
    items: Collection[T],
    /,
    num: int,
    *,
    chunks: Mapping[K, SupportsIndex],
    rng: int | Random | None = None,
    group_labels: Collection[Label] | None = None,
    max_error: float | None = None,
) -> list[dict[K, list[T]]]: ...
@overload
def rotating_partitions[T](
    items: Collection[T],
    /,
    num: int,
    *,
    chunks: Sequence[SupportsIndex],
    rng: int | Random | None = None,
    group_labels: Collection[Label] | None = None,
    max_error: float | None = None,
) -> list[tuple[list[T], ...]]: ...
def rotating_partitions[T, K](
    items: Collection[T],
    /,
    num: int,
    *,
    chunks: Mapping[K, SupportsIndex] | Sequence[SupportsIndex],
    rng: int | Random | None = None,
    group_labels: Collection[Label] | None = None,
    max_error: float | None = None,
) -> list[dict[K, list[T]]] | list[tuple[list[T], ...]]:
    r"""Partition collection elements by rotating optionally shuffled group chunks.

    By default, retain distinct groups in first-occurrence order. Supplying `rng`
    shuffles groups once before forming chunks. Distribute groups
    over the configured total number of chunks; chunk sizes differ by at most one,
    and earlier chunks receive leftovers.
    Fold `i` rotates split boundaries by `i * num_chunks // num` chunks.
    The number of folds must not exceed the total chunk count, ensuring distinct
    rotation offsets. Circular gaps between offsets differ by at most one chunk.
    With `num=1`, return a one-element list containing the initial partition.

    Splits assigned zero chunks are retained as empty lists. Too few
    groups can produce empty splits, including training splits. Group counts,
    rather than element counts, determine the proportions when grouping is used.
    Splits can overlap across folds despite distinct rotation offsets.

    Across `num` folds, every chunk visits a split with `count` chunks either
    `floor(num * count / num_chunks)` or `ceil(num * count / num_chunks)` times.
    This balances coverage between chunks to within one visit, with exact equality
    whenever `num * count` is divisible by `num_chunks`.

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
        num: Required positive number of folds, at most the total chunk count.
        chunks: Mapping from split keys to non-negative integer chunk counts, or
            a sequence of counts. Counts must support `__index__`.
            Their positive sum must be at least `num`.
            Mapping keys are preserved exactly without normalization.
        rng: Optional integer seed or existing `random.Random` instance to shuffle
            groups before forming chunks. An integer creates a local RNG; an
            existing instance advances in place. Defaults to `None`, which
            disables shuffling and preserves first-occurrence order.
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

    See Also:
        - [`KFold`][sklearn.model_selection.KFold]: Standard train/test k-fold
          splitting; the special case `chunks={"train": num - 1, "test": 1}`.
        - [`GroupKFold`][sklearn.model_selection.GroupKFold]: K-fold splitting
          with intact groups. With `shuffle=True`, allocation uses group counts.
        - [`RepeatedKFold`][sklearn.model_selection.RepeatedKFold]: Repeats full
          k-fold cycles with new randomization for each cycle.
        - [`random_partitions`][tsdm.tasks.utils.random_partitions]: Independent
          random partitions without coordinated rotation.

    Examples:
        Rotate training and test splits over two folds for four elements.

        ```pycon
        >>> items = [0, 1, 2, 3]
        >>> rotating_partitions(items, num=2, chunks=[1, 1])
        [([0, 1], [2, 3]), ([2, 3], [0, 1])]
        >>> rotating_partitions(items, num=2, chunks=[1, 1], rng=0)
        [([0, 2], [1, 3]), ([1, 3], [0, 2])]
        >>> rotating_partitions(items, num=2, chunks={"train": 1, "test": 1}, rng=0)
        [{'train': [0, 2], 'test': [1, 3]}, {'train': [1, 3], 'test': [0, 2]}]

        ```
    """
    _validate_args(items, num, group_labels=group_labels)
    counts = _normalize_counts(chunks, name="chunks")
    num_chunks = sum(counts)

    # More folds than chunks would repeat rotation offsets.
    if num > num_chunks:
        raise ValueError("The number of folds must not exceed the total chunk count.")

    # Deduplicate groups in element order, then optionally shuffle once for all folds.
    labels = list(range(len(items))) if group_labels is None else list(group_labels)
    unique_groups = list(dict.fromkeys(labels))
    if rng is not None:
        generator = _normalize_rng(rng)
        generator.shuffle(unique_groups)

    # Divide the groups into chunks. The first few chunks receive one
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
    # Later folds distribute rotation offsets as evenly as possible around the cycle.
    chunk_splits = [split for split, count in enumerate(counts) for _ in range(count)]
    result: list[tuple[list[T], ...]] = []

    for fold in range(num):
        offset = fold * num_chunks // num
        # Retain empty splits, then rotate chunk boundaries and collect
        # items in their original order in one pass.
        parts: tuple[list[T], ...] = tuple([] for _ in counts)
        for item, label in zip(items, labels, strict=True):
            chunk = chunk_by_group[label]
            split = chunk_splits[(chunk - offset) % num_chunks]
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
    partition_union = {*p for p in partition}  # pyrefly: ignore[not-a-type]
    if partition_union != set(union):
        return False
    return len(partition_union) == sum(len(p) for p in partition)


@overload
def random_partitions[T, K](
    items: Collection[T],
    /,
    num: int,
    *,
    sizes: Mapping[K, SupportsIndex],
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: float | None = None,
) -> list[dict[K, list[T]]]: ...
@overload
def random_partitions[T, K](
    items: Collection[T],
    /,
    num: int,
    *,
    ratios: Mapping[K, SupportsFloat],
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: float | None = None,
) -> list[dict[K, list[T]]]: ...
@overload
def random_partitions[T](
    items: Collection[T],
    /,
    num: int,
    *,
    sizes: Sequence[SupportsIndex],
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: float | None = None,
) -> list[tuple[list[T], ...]]: ...
@overload
def random_partitions[T](
    items: Collection[T],
    /,
    num: int,
    *,
    ratios: Sequence[SupportsFloat],
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: float | None = None,
) -> list[tuple[list[T], ...]]: ...
def random_partitions[T, K](
    items: Collection[T],
    /,
    num: int,
    *,
    sizes: Mapping[K, SupportsIndex] | Sequence[SupportsIndex] | None = None,
    ratios: Mapping[K, SupportsFloat] | Sequence[SupportsFloat] | None = None,
    rng: int | Random,
    group_labels: Collection[Label] | None = None,
    max_error: float | None = None,
) -> list[dict[K, list[T]]] | list[tuple[list[T], ...]]:
    r"""Draw `num` random partitions of indivisible groups of items.

    Group labels can mark redundant samples, such as results from replication
    experiments. Samples with the same label always share a split to avoid data
    leakage. These labels keep groups together; they do not stratify classes
    across splits. Without labels, each item occurrence forms its own group.
    Duplicates and unhashable items are allowed, and every occurrence is retained.
    Pass `group_labels=items` to keep equal hashable items together, or supply
    hashable labels for unhashable items. Labels need not be sortable.

    Exactly one of `sizes` or `ratios` must be provided. Both count groups, not
    samples. Allocation proceeds as follows:

    1. Collect distinct group labels in first-occurrence order.
    2. Use the exact group counts in `sizes`, or compute quotas from the number
       of groups and normalized `ratios`. Floor the quotas, then allocate leftover
       groups by largest fractional remainder. Ties favor earlier splits in the
       supplied sequence or mapping. For example, five groups with ratios
       `[1, 1, 1]` receive counts `[2, 2, 1]`.
    3. For each partition, shuffle a fresh copy of the groups and assign
       consecutive groups using these counts.
       All assignments with the resulting group counts are equally likely.
    4. Collect items in input order within each split, including interleaved
       occurrences from different groups.

    Group sizes never influence allocation. Sample proportions can differ
    substantially from group proportions. Zero counts retain empty splits, and
    even positive ratios can yield empty splits: two groups with train/test
    ratios of 9:1 have quotas of 1.8 and 0.2, so both go to training.

    `max_error` bounds the total variation distance between actual sample
    proportions and normalized `ratios`. It checks each drawn partition and raises
    on failure, without retries or optimization. Empty inputs have no sample
    distribution and fail this check. With explicit `sizes`, the check is ignored.

    One RNG advances across all draws; it is never reseeded between partitions.
    An integer seed reproduces the full sequence without changing global random
    state. An existing `random.Random` instance also retains its state between
    calls. Draws can overlap or coincide; they do not guarantee rotation coverage.

    Args:
        items: Ordered collection of items. Duplicates and unhashable items are
            allowed. Unordered sets are rejected.
        num: Required positive number of partitions to draw.
        sizes: Mapping from split keys to exact non-negative group counts, or a
            sequence of counts. Counts must support `__index__` and sum to the
            number of distinct groups (the number of items without labels).
            At least one count must be positive. Mutually exclusive with `ratios`.
        ratios: Mapping from split keys to relative group weights, or a sequence
            of weights. Weights must be finite, non-negative, and not all zero.
            Mutually exclusive with `sizes`.
        rng: Required integer seed or existing `random.Random` instance. An
            integer creates a local RNG; an existing instance advances in place.
        group_labels: Optional ordered collection of stable, hashable labels,
            aligned with `items` by iteration order and of the same length.
            Occurrences with equal labels always stay together.
        max_error: Maximum sample-proportion TV distance from `ratios`, in `[0, 1]`.
            `None` disables checking. Ignored with explicit `sizes`.

    Returns:
        A list of `num` partitions, including when `num=1`. Each partition
        is a dictionary preserving the exact keys and order of a supplied mapping,
        or a tuple preserving the length and order of a supplied sequence.
        Each split contains a list of items in input order. Empty splits and
        every item occurrence are retained. With `ratios`, empty inputs yield
        empty splits when checking is disabled.

    Raises:
        ValueError: Invalid `num`, sizes, ratios, or input lengths; invalid `max_error`
            with ratios; or an error check on empty inputs.
        TypeError: Unordered inputs, unhashable group labels, or an invalid RNG.
        RuntimeError: Sample proportions exceed `max_error`.

    See Also:
        - [`ShuffleSplit`][sklearn.model_selection.ShuffleSplit]: Repeated random
          train/test splits without grouping.
        - [`GroupShuffleSplit`][sklearn.model_selection.GroupShuffleSplit]:
          Repeated random train/test splits with intact groups; sizes count groups.
        - [`train_test_split`][sklearn.model_selection.train_test_split]: A single
          train/test split, optionally stratified by class labels.
        - [`rotating_partitions`][tsdm.tasks.utils.rotating_partitions]: Partitions
          coordinated by rotating chunks after one shuffle.

    Examples:
        Specify exact counts or named relative weights.

        ```pycon
        >>> [partition] = random_partitions(range(4), num=1, sizes=[2, 2], rng=0)
        >>> partition
        ([0, 2], [1, 3])
        >>> [partition] = random_partitions(
        ...     range(4), num=1, ratios={"train": 1, "test": 1}, rng=0
        ... )
        >>> partition
        {'train': [0, 2], 'test': [1, 3]}

        ```

        Items with the same group label count as one unit: "a1" and "a2" form
        one group, so `sizes=[1, 2]` allocates three groups containing four items.

        ```pycon
        >>> # "a1" and "a2" count as one unit because they share a group label.
        >>> [partition] = random_partitions(
        ...     ["a1", "b1", "a2", "c1"],
        ...     num=1,
        ...     sizes=[1, 2],
        ...     group_labels=["a", "b", "a", "c"],
        ...     rng=0,
        ... )
        >>> partition
        (['a1', 'a2'], ['b1', 'c1'])

        ```

        Draw several partitions with one advancing RNG.

        ```pycon
        >>> random_partitions(range(4), num=2, sizes=[2, 2], rng=2022)
        [([0, 3], [1, 2]), ([1, 3], [0, 2])]

        ```
    """
    _validate_args(items, num, group_labels=group_labels)
    rng = _normalize_rng(rng)
    labels = list(range(len(items))) if group_labels is None else list(group_labels)
    unique_groups = list(dict.fromkeys(labels))
    num_groups = len(unique_groups)
    weights: list[float] = []

    match sizes, ratios:
        case [sizes, None] if sizes is not None:
            counts = _normalize_counts(sizes, name="sizes")
            if sum(counts) != num_groups:
                raise ValueError("sizes must sum to the number of groups")
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
            quotas = [num_groups * weight / total for weight in normalized]
            counts = [floor(quota) for quota in quotas]

            # Stable descending order gives earlier splits precedence on ties.
            if remainder := num_groups - sum(counts):
                order = sorted(
                    range(len(counts)),
                    key=lambda i: quotas[i] - counts[i],
                    reverse=True,
                )
                for index in order[:remainder]:
                    counts[index] += 1
        case _:
            raise ValueError("Exactly one of sizes or ratios must be provided")

    result: list[tuple[list[T], ...]] = []
    for _ in range(num):
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        assignments: dict[Label, int] = {}
        start = 0
        for split, count in enumerate(counts):
            stop = start + count
            assignments.update(dict.fromkeys(shuffled[start:stop], split))
            start = stop

        # Membership is random; collecting in one pass retains input order per split.
        partitions: list[list[T]] = [[] for _ in counts]
        for item, label in zip(items, labels, strict=True):
            partitions[assignments[label]].append(item)
        if ratios is not None and max_error is not None:
            _check_partition_error(
                [len(part) for part in partitions], weights, max_error=max_error
            )
        result.append(tuple(partitions))

    configuration = sizes if sizes is not None else ratios
    if isinstance(configuration, Mapping):
        return [dict(zip(configuration, parts, strict=True)) for parts in result]
    return result
