r"""Utility functions for linear algebra operations."""

__all__ = [
    "get_broadcast",
]


from tsdm.types.aliases import Axis


def get_broadcast(
    original_shape: tuple[int, ...],
    /,
    *,
    axis: Axis,
    keepdim: bool = False,
) -> tuple[slice | None, ...]:
    r"""Create an indexer that restores axes removed by a tensor contraction.

    If `x = contraction(data, axis=axis)`, then `x[get_broadcast(data.shape, axis=axis)]`
    has the same shape as `contraction(data, axis=axis, keepdims=True)`.
    This allows element-wise operations such as `data + x[broadcast]`.

    With `keepdim=True`, `axis` instead selects the axes retained by the
    contraction. Negative axes are normalized modulo the rank.

    Args:
        original_shape: The shape of the tensor before contraction.
        axis: The axes to contract, or `None` to select all axes.
        keepdim: Whether `axis` selects retained axes instead of contracted axes.

    Example:
        ```pycon
        >>> import numpy as np
        >>> arr = np.ones([1, 2, 3, 4, 5])
        >>> ax = (1, -1)
        >>> broadcast = get_broadcast(arr.shape, axis=ax)
        >>> m = np.mean(arr, axis=ax)
        >>> m_ref = np.mean(arr, axis=ax, keepdims=True)
        >>> m[broadcast].shape == m_ref.shape
        True
        ```

        For shape `(2, 3, 4, 5, 6, 7)` and contracted axes `(0, 2, -1)`,
        the indexer is `(None, slice(None), None, slice(None), slice(None), None)`.
    """
    rank = len(original_shape)

    if keepdim:
        match axis:
            case None:  # all axes are contracted
                kept_axis = set()
            case int():
                kept_axis = {axis % rank}
            case _:
                kept_axis = {a % rank for a in axis}
        return tuple(slice(None) if a in kept_axis else None for a in range(rank))

    match axis:
        case None:  # all axes are contracted
            contracted_axes = set(range(rank))
        case int():
            contracted_axes = {axis % rank}
        case _:
            contracted_axes = {a % rank for a in axis}

    return tuple(None if a in contracted_axes else slice(None) for a in range(rank))
