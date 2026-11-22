r"""NumPy distances between finite probability distributions.

Inputs have shape `(..., n)`, where the nonempty last axis contains finite real
weights for the same `n` categories. Leading batch dimensions are broadcast
together. Each distribution is normalized independently, so counts and relative
sizes can be supplied directly. Every distribution must have positive total
weight. Category positions must correspond; these functions do not align labels
or infer a ground metric. Empty batches are supported.

Fisher–Rao, Hellinger, and total variation allow zero weights. Functions return
float64 NumPy arrays with the broadcast batch shape, or NumPy scalars for
unbatched inputs, and do not modify their inputs.

Throughout, $n$ denotes the number of coordinates, $eᵢ$ a simplex vertex, and
$uₙ=(1/n, …, 1/n)$ the center. Distinct vertices exist only for $n≥2$.

| Metric          | Distinct vertices $d(eᵢ,eⱼ)$ | Vertex to center $d(eᵢ,uₙ)$ |
|-----------------|------------------------------|-----------------------------|
| Fisher–Rao      | $π$                          | $2\arccos(1/√n)$            |
| Hellinger       | $1$                          | $\sqrt{1-1/√n}$             |
| Total variation | $1$                          | $1-1/n$                     |
"""

__all__ = [
    "fisher_rao_distance",
    "hellinger_distance",
    "total_variation_distance",
]

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_weights(
    p: ArrayLike, q: ArrayLike, /
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Validate category counts, broadcast batch shapes, and weight domains."""
    raw_p, raw_q = np.asarray(p), np.asarray(q)
    if np.iscomplexobj(raw_p) or np.iscomplexobj(raw_q):
        raise ValueError("Weights must be real.")
    x, y = np.asarray(raw_p, dtype=np.float64), np.asarray(raw_q, dtype=np.float64)
    if x.ndim == 0 or y.ndim == 0 or x.shape[-1] == 0 or x.shape[-1] != y.shape[-1]:
        raise ValueError("Weights must have matching nonempty category axes.")
    try:
        np.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    except ValueError as exc:
        raise ValueError("Weights must have broadcastable batch shapes.") from exc
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Weights must be finite.")
    if (
        np.any(x < 0)
        or np.any(y < 0)
        or not np.all(np.any(x > 0, axis=-1))
        or not np.all(np.any(y > 0, axis=-1))
    ):
        raise ValueError("Weights must be nonnegative with positive total weight.")
    return x, y


def _normalize(x: NDArray[np.float64], /) -> NDArray[np.float64]:
    r"""Normalize nonnegative weights without overflowing their sum."""
    scaled = x / x.max(axis=-1, keepdims=True)
    return scaled / scaled.sum(axis=-1, keepdims=True)


def fisher_rao_distance(p: ArrayLike, q: ArrayLike, /) -> NDArray[np.float64]:
    r"""Compute the Fisher–Rao geodesic distance between probability vectors.

    For independently normalized inputs $p$ and $q$, use the radius-two convention:

    $$ d_{FR}(p,q) = 2\arccos\Bigl(∑ᵢ\sqrt{pᵢqᵢ}\Bigr) ∈ [0,π] $$

    The square-root embedding $p↦2√p$ lies on a sphere of radius two.
    Distinct vertices have square-root inner product zero, while a vertex and
    the center have inner product $1/√n$. Consequently,

    $$ d_{FR}(eᵢ,eⱼ) = π \quad (i≠j), \qquad d_{FR}(eᵢ,uₙ) = 2\arccos(1/√n) $$

    The implementation uses the equivalent chord formula

    $$ d_{FR}(p,q) = 4\operatorname{atan2}(‖√p - √q‖₂,‖√p + √q‖₂)$$

    to avoid cancellation in arccos near identical distributions.

    Args:
        p: Nonnegative finite real weights of shape `(..., n)`, with `n > 0`
            and positive total weight along the last axis for every distribution.
        q: Weights for the same `n` categories, with broadcastable batch dimensions.

    Returns:
        Float64 distances with the broadcast batch shape. Unbatched inputs
        produce a NumPy scalar. The category axis is reduced.

    Raises:
        ValueError: If the shapes or weights are invalid.

    Examples:
        Relative sizes need not sum to one:

        >>> fisher_rao_distance([3, 1, 1], [4, 2, 1]).round(4).item()
        0.2261
        >>> fisher_rao_distance([1, 2], [2, 4]).round(4).item()
        0.0
        >>> fisher_rao_distance([1, 0, 0], [0, 1, 0]).round(4).item()
        3.1416
        >>> n = 4
        >>> vertex, center = np.eye(n)[0], np.full(n, 1 / n)
        >>> fisher_rao_distance(vertex, center).round(4).item()
        2.0944
        >>> fisher_rao_distance([[1, 0], [1, 1]], [0, 1]).round(4)
        array([3.1416, 1.5708])
    """
    x, y = _validate_weights(p, q)
    x = np.sqrt(_normalize(x))
    y = np.sqrt(_normalize(y))
    return 4 * np.arctan2(
        np.linalg.vector_norm(x - y, axis=-1),
        np.linalg.vector_norm(x + y, axis=-1),
    )


def hellinger_distance(p: ArrayLike, q: ArrayLike, /) -> NDArray[np.float64]:
    r"""Compute the normalized Hellinger distance between probability vectors.

    For independently normalized inputs $p$ and $q$,

    $$ H(p,q) = \frac{‖√p - √q‖₂}{√2} = \sqrt{1-∑ᵢ\sqrt{pᵢqᵢ}} ∈ [0,1] $$

    Distinct vertices have disjoint support and hence square-root inner product
    zero. For a vertex and the center the inner product is $1/\sqrt n$, giving

    $$ H(eᵢ,eⱼ) = 1 \quad (i≠j), \qquad H(eᵢ,uₙ) = \sqrt{1-1/√n} $$

    This convention includes the factor $1/\sqrt 2$; the squared Hellinger
    distance is $H²$. Its relation to Fisher–Rao is $H=\sqrt 2\sin(d_{FR}/4)$.
    The implementation uses the square-root difference to retain accuracy near
    identical distributions.

    Args:
        p: Nonnegative finite real weights of shape `(..., n)`, with `n > 0`
            and positive total weight along the last axis for every distribution.
        q: Weights for the same `n` categories, with broadcastable batch dimensions.

    Returns:
        Float64 distances with the broadcast batch shape. Unbatched inputs
        produce a NumPy scalar. The category axis is reduced.

    Raises:
        ValueError: If the shapes or weights are invalid.

    Examples:
        >>> hellinger_distance([3, 1, 1], [4, 2, 1]).round(4).item()
        0.0799
        >>> hellinger_distance([1, 2], [2, 4]).round(4).item()
        0.0
        >>> hellinger_distance([1, 0, 0], [0, 1, 0]).round(4).item()
        1.0
        >>> n = 4
        >>> vertex, center = np.eye(n)[0], np.full(n, 1 / n)
        >>> hellinger_distance(vertex, center).round(4).item()
        0.7071
        >>> hellinger_distance([[1, 0], [1, 1]], [0, 1]).round(4)
        array([1.    , 0.5412])
    """
    x, y = _validate_weights(p, q)
    x = np.sqrt(_normalize(x))
    y = np.sqrt(_normalize(y))
    return np.linalg.vector_norm(x - y, axis=-1) / np.sqrt(2)


def total_variation_distance(p: ArrayLike, q: ArrayLike, /) -> NDArray[np.float64]:
    r"""Compute the total variation distance between probability vectors.

    For independently normalized inputs $p$ and $q$,

    $$ d_{TV}(p,q) = ½∑ᵢ|pᵢ-qᵢ| ∈ [0,1] $$

    Distinct vertices differ by one in two coordinates. A vertex differs from
    the center by $1-1/n$ in its occupied coordinate and by $1/n$ in each of the
    other $n-1$ coordinates. Thus,

    $$ d_{TV}(eᵢ,eⱼ) = 1 \quad (i≠j), \qquad d_{TV}(eᵢ,uₙ) = 1-1/n $$

    For split proportions this measures the mass that must be reassigned,
    ignoring grouping and integer-count constraints. For at most three
    categories, it equals the maximum absolute difference of proportions.

    Args:
        p: Nonnegative finite real weights of shape `(..., n)`, with `n > 0`
            and positive total weight along the last axis for every distribution.
        q: Weights for the same `n` categories, with broadcastable batch dimensions.

    Returns:
        Float64 distances with the broadcast batch shape. Unbatched inputs
        produce a NumPy scalar. The category axis is reduced.

    Raises:
        ValueError: If the shapes or weights are invalid.

    Note:
        Total variation is the Wasserstein-1 distance for the discrete ground
        metric $ρ(i,j)=0$ if $i=j$ and $ρ(i,j)=1$ otherwise. For normalized
        inputs, let $Π(p,q)$ be the nonnegative transport matrices with row sums
        $p$ and column sums $q$. Moving mass between distinct categories costs
        one per unit, while mass retained on the diagonal costs zero. Hence,

        $$\begin{aligned}
            W₁(p,q) &= \min_{γ∈Π(p,q)} ∑_{i≠j} γᵢⱼ \\
                    &= 1-∑ᵢ\min(pᵢ,qᵢ) = ½∑ᵢ|pᵢ-qᵢ| = d_{TV}(p,q)
        \end{aligned}$$

        The optimum keeps $\min(pᵢ,qᵢ)$ at each category and transports the
        remaining mass from categories with excess to those with deficits.

    Examples:
        >>> total_variation_distance([3, 1, 1], [8, 1, 1]).round(10).item()
        0.2
        >>> total_variation_distance([1, 2], [2, 4]).item()
        0.0
        >>> total_variation_distance([1, 0, 0], [0, 1, 0]).item()
        1.0
        >>> n = 4
        >>> vertex, center = np.eye(n)[0], np.full(n, 1 / n)
        >>> total_variation_distance(vertex, center).item()
        0.75
        >>> total_variation_distance([[1, 0], [1, 1]], [0, 1])
        array([1. , 0.5])
    """
    x, y = _validate_weights(p, q)
    x = _normalize(x)
    y = _normalize(y)
    return 0.5 * np.linalg.vector_norm(x - y, axis=-1, ord=1)
