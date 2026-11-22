r"""Reproducible partitions of time-series collection elements.

Identifiers refer to complete collection elements, never observation rows or
sample windows. Optional group labels keep several elements in the same split.
Split proportions apply to groups when supplied, otherwise to individual elements.

Both fold generators partition their input exactly once per fold, retain empty
splits, and preserve input order within each split. Reproducibility requires
the same seed or initial RNG state, ordered identifiers, split configuration,
group labels, and Python random implementation. Unordered sets of identifiers
are rejected by the fold generators. RNGs constructed from integer seeds do not
affect global random state. These stdlib algorithms do not reproduce historical
NumPy assignments; persist assignments when exact reproduction across algorithm
or runtime changes is required.

Fold keys end in a `SplitType` and support any number of preceding fold indices.
A `fold_prefix` only namespaces keys: for nested cross-validation, explicitly
pass the outer training identifiers to the inner generator to prevent leakage.
"""

__all__ = [
    "FoldKey",
    "Folds",
    "repeated_random_folds",
    "rotating_folds",
    "random_partition",
    "is_partition",
]

from collections.abc import Collection, Iterable, Mapping, Sequence, Set as AbstractSet
from math import floor, fsum, isfinite
from random import Random
from typing import Any, Optional, SupportsFloat, SupportsIndex, overload

from .base import SplitType

type FoldKey = tuple[*tuple[int, ...], SplitType]
type Folds[K] = dict[FoldKey, list[K]]


def _normalize_rng(rng: int | Random, /) -> Random:
    r"""Create a local RNG from a seed or reuse an existing instance unchanged."""
    match rng:
        case Random():
            return rng
        case int():
            return Random(rng)
        case _:
            raise TypeError("rng must be an integer seed or a random.Random instance.")


def _normalize_weights[StringT: str, V: float](
    values: Mapping[StringT, V], /
) -> dict[SplitType, V]:
    r"""Normalize split names, preserving values and insertion order.

    Reject empty configurations and aliases that identify the same split type.
    """
    normalized = {SplitType(name): value for name, value in values.items()}

    # validate the result
    if not normalized:
        raise ValueError("At least one split must be configured.")
    if len(normalized) != len(values):
        raise ValueError("Split aliases must identify distinct split types.")
    if any(not isfinite(weight) for weight in normalized.values()):
        raise ValueError("Split weights must be finite.")
    if any(weight < 0 for weight in normalized.values()):
        raise ValueError("Split weights must be non-negative.")
    if not any(weight > 0 for weight in normalized.values()):
        raise ValueError("At least one split must have a positive weight.")

    return normalized


def _validate_args(keys: Collection, groups: Mapping | None, num_folds: int) -> None:
    r"""Validate keys."""
    if not isinstance(num_folds, int) or num_folds <= 0:
        raise ValueError("num_folds must be a positive integer.")
    if isinstance(keys, AbstractSet):
        raise TypeError("keys must have a reproducible iteration order, not a set.")
    if len(set(keys)) != len(keys):
        raise ValueError("Collection identifiers must be unique.")
    if groups is not None and set(groups) != set(keys):
        raise ValueError("groups must map exactly the supplied collection identifiers.")


def rotating_folds[K](
    keys: Collection[K],
    /,
    chunks: Mapping[Any, int],  # Mapping[str & Any, int]
    *,
    num_folds: int,
    rng: int | Random,
    fold_prefix: Sequence[int] = (),
    groups: Mapping[K, object] | None = None,
) -> Folds[K]:
    r"""Partition collection elements by rotating shuffled group chunks.

    Shuffle distinct groups once and distribute them over `sum(chunks.values())`
    chunks. Chunk sizes differ by at most one; earlier chunks receive leftovers.
    Each fold shifts split boundaries by `num_chunks // num_folds` chunks.
    This preserves the rotation policy of `datatools.folds_from_groups`.

    Args:
        keys: Collection of unique, ordered identifiers of complete elements.
        chunks: Ordered split names or aliases mapped to non-negative integer
            chunk counts. Their positive sum must be divisible by `num_folds`.
        num_folds: Positive number of folds.
        rng: Required integer seed or existing `random.Random` instance. An
            integer creates a local RNG; an existing instance advances in place.
        groups: Optional mapping from every identifier to a stable group label.
        fold_prefix: Non-negative outer fold indices prepended to generated keys.

    Splits assigned zero chunks and empty inputs are retained as empty lists. Too few
    groups can produce empty splits, including training splits. Group counts,
    rather than element counts, determine the proportions when grouping is used.
    A one-chunk test split with `num_folds` chunks visits every group once;
    arbitrary chunk counts do not guarantee equal coverage across folds.

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
        {(0, <SplitType.TRAIN: 'train'>): [0, 3],
         (0, <SplitType.TEST: 'test'>): [1, 2],
         (1, <SplitType.TRAIN: 'train'>): [1, 2],
         (1, <SplitType.TEST: 'test'>): [0, 3]}

        ```
    """
    _validate_args(keys, groups=groups, num_folds=num_folds)
    rng = _normalize_rng(rng)
    chunks = _normalize_weights(chunks)
    num_chunks = sum(chunks.values())

    if not all(isinstance(count, int) for count in chunks.values()):
        raise ValueError("chunks must contain non-negative integer counts.")
    if num_chunks % num_folds:
        raise ValueError("Total chunk count must divisible by num_folds.")

    # Deduplicate groups in element order, then shuffle once for all folds.
    labels = (
        {key: key for key in keys}
        if groups is None
        else {key: groups[key] for key in keys}
    )
    unique_groups = list(dict.fromkeys(labels.values()))
    rng.shuffle(unique_groups)

    # Divide the shuffled groups into chunks. The first few chunks receive one
    # extra group when the group count is not divisible by the chunk count.
    base_size, num_larger_chunks = divmod(len(unique_groups), num_chunks)
    chunk_by_group: dict[object, int] = {}
    start = 0

    for chunk in range(num_chunks):
        chunk_size = base_size + (1 if chunk < num_larger_chunks else 0)
        stop = start + chunk_size
        chunk_by_group.update(dict.fromkeys(unique_groups[start:stop], chunk))
        start = stop

    # Each split reserves its configured number of consecutive chunks in fold zero.
    # Later folds move these boundaries by a fixed number of chunks.
    chunk_splits = [split for split, count in chunks.items() for _ in range(count)]
    step = num_chunks // num_folds
    result: Folds[K] = {}

    for fold in range(num_folds):
        # Retain empty splits, then rotate chunk boundaries and collect
        # identifiers in their original order in one pass.
        result.update({(*fold_prefix, fold, split): [] for split in chunks})  # pyrefly: ignore[no-matching-overload]
        for key, label in labels.items():
            chunk = chunk_by_group[label]
            split = chunk_splits[(chunk - fold * step) % num_chunks]
            result[(*fold_prefix, fold, split)].append(key)

    return result


def repeated_random_folds[K](
    keys: Collection[K],
    /,
    ratios: Mapping[Any, float],  # Mapping[str & Any, float]
    *,
    num_folds: int,
    rng: int | Random,
    groups: Optional[Mapping[K, object]] = None,
    fold_prefix: Sequence[int] = (),
) -> Folds[K]:
    r"""Create repeated random partitions of collection elements.

    Args:
        keys: Collection of unique, ordered identifiers of complete elements.
        ratios: Ordered split names or aliases mapped to finite, non-negative
            relative proportions, with at least one positive ratio. Allocate group
            counts by largest remainder, breaking ties by split insertion order.
        num_folds: Positive number of partitions to draw.
        rng: Required integer seed or existing `random.Random` instance. One RNG
            advances across folds; an existing instance also retains its state
            between calls. Partitions may coincide by chance for small inputs.
        groups: Optional exact identifier-to-group mapping. Labels must be stable
            and hashable.
        fold_prefix: Non-negative outer fold indices prepended to generated keys.

    These are repeated holdouts: test sets can overlap across folds. No rotation
    coverage is guaranteed. Empty inputs and zero-ratio splits yield empty lists.
    Group proportions need not equal collection-element proportions.

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
        {(0, <SplitType.TRAIN: 'train'>): [0, 3],
         (0, <SplitType.TEST: 'test'>): [1, 2],
         (1, <SplitType.TRAIN: 'train'>): [1, 3],
         (1, <SplitType.TEST: 'test'>): [0, 2]}

        ```
    """
    _validate_args(keys, groups=groups, num_folds=num_folds)
    rng = _normalize_rng(rng)
    ratios = _normalize_weights(ratios)
    scale = max(ratios.values())

    # Count distinct groups in element order. Without grouping, each element
    # contributes one group, so the same allocation procedure applies.
    labels = (
        {key: key for key in keys}
        if groups is None
        else {key: groups[key] for key in keys}
    )
    unique_groups = list(dict.fromkeys(labels.values()))

    # Scale ratios before summing to avoid overflow for large finite ratios.
    normalized = [ratio / scale for ratio in ratios.values()]
    total = fsum(normalized)
    quotas = [len(unique_groups) * ratio / total for ratio in normalized]
    counts = [floor(quota) for quota in quotas]

    # Allocate leftover groups to the largest fractional quotas. Python's stable
    # sort breaks ties by the insertion order of the configured splits.
    remainder_order = sorted(
        range(len(ratios)), key=lambda i: quotas[i] - counts[i], reverse=True
    )
    for index in remainder_order[: len(unique_groups) - sum(counts)]:
        counts[index] += 1

    # Advance one local RNG across folds instead of reseeding it for each fold.
    result: Folds[K] = {}

    for fold in range(num_folds):
        # Shuffle a fresh copy so each draw starts from the same group order.
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)

        # Assign consecutive groups to splits using the precomputed counts.
        assignments: dict[object, SplitType] = {}
        start = 0
        for split, count in zip(ratios, counts, strict=True):
            stop = start + count
            assignments.update(dict.fromkeys(shuffled[start:stop], split))
            start = stop

        # Retain empty splits, then collect identifiers in one pass to preserve
        # input order without scanning all elements separately for each split.
        result.update({(*fold_prefix, fold, split): [] for split in ratios})  # pyrefly: ignore[no-matching-overload]
        for key, label in labels.items():
            split = assignments[label]
            result[(*fold_prefix, fold, split)].append(key)

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
def random_partition[K](
    keys: Collection[K],
    *,
    sizes: Sequence[SupportsIndex],
    rng: int | Random,
) -> tuple[list[K], ...]: ...
@overload
def random_partition[K](
    keys: Collection[K],
    *,
    ratios: Sequence[SupportsFloat],
    rng: int | Random,
) -> tuple[list[K], ...]: ...
def random_partition[K](
    keys: Collection[K],
    *,
    sizes: Sequence[SupportsIndex] | None = None,
    ratios: Sequence[SupportsFloat] | None = None,
    rng: int | Random,
) -> tuple[list[K], ...]:
    r"""Create a random partition for a dataset.

    Exactly one of `sizes` or `ratios` must be provided. Sizes are absolute
    partition counts and must sum to the number of keys. Ratios are normalized
    weights used to distribute all keys between the partitions.

    Args:
        keys: Keys to distribute among the partitions.
        sizes: Absolute partition counts. The counts must be non-negative and
            sum to the number of keys. Mutually exclusive with `ratios`.
        ratios: Relative partition weights. The weights must be finite, non-negative
            and not all zero. Mutually exclusive with `sizes`.
        rng: Required integer seed or existing `random.Random` instance. An
            integer creates a local RNG; an existing instance advances in place.

    Returns:
        A tuple of key lists in shuffled order, one per requested partition.
    """
    rng = _normalize_rng(rng)
    num = len(keys)

    match sizes, ratios:
        case [sizes, None] if sizes is not None:
            counts = [size.__index__() for size in sizes]
            if any(count < 0 for count in counts) or sum(counts) != num:
                raise ValueError(
                    "sizes must be non-negative and sum to the number of keys"
                )
        case [None, ratios] if ratios is not None:
            weights = [ratio.__float__() for ratio in ratios]

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
        case _:
            raise ValueError("Exactly one of sizes or ratios must be provided")

    shuffled = list(keys)
    rng.shuffle(shuffled)
    partitions = []
    start = 0
    for count in counts:
        stop = start + count
        partitions.append(shuffled[start:stop])
        start = stop
    return tuple(partitions)
