r"""Check simplex geometry, weight handling, and numerical edge cases."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from tsdm.stats.distances import (
    fisher_rao_distance,
    hellinger_distance,
    total_variation_distance,
)

type Distance = Callable[[ArrayLike, ArrayLike], NDArray[np.float64]]

DISTANCES = (fisher_rao_distance, hellinger_distance, total_variation_distance)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 100])
def test_simplex_geometry(n: int) -> None:
    r"""Check the vertex and center formulas, including the singleton simplex."""
    vertex, center = np.eye(n)[0], np.full(n, 1 / n)
    assert fisher_rao_distance(vertex, center) == pytest.approx(
        2 * np.arccos(1 / np.sqrt(n))
    )
    assert hellinger_distance(vertex, center) == pytest.approx(
        np.sqrt(1 - 1 / np.sqrt(n))
    )
    assert total_variation_distance(vertex, center) == pytest.approx(1 - 1 / n)

    if n > 1:
        other = np.eye(n)[1]
        assert fisher_rao_distance(vertex, other) == pytest.approx(np.pi)
        assert hellinger_distance(vertex, other) == pytest.approx(1)
        assert total_variation_distance(vertex, other) == pytest.approx(1)


@pytest.mark.parametrize("distance", DISTANCES)
def test_metric_properties(distance: Distance) -> None:
    r"""Check identity, symmetry, and the triangle inequality on boundary points."""
    p, q, r = [3, 0, 1, 0], [0, 1, 2, 3], [1, 1, 1, 1]
    assert distance(p, p) == 0
    assert distance(p, q) > 0
    assert distance(p, q) == pytest.approx(distance(q, p))
    assert distance(p, q) <= distance(p, r) + distance(r, q)


@pytest.mark.parametrize("distance", DISTANCES)
def test_weight_normalization(distance: Distance) -> None:
    r"""Counts, probabilities, and extreme finite scales give the same distance."""
    p, q = np.array([3.0, 0.0, 1.0]), np.array([4.0, 2.0, 1.0])
    original_p, original_q = p.copy(), q.copy()
    expected = distance(p / p.sum(), q / q.sum())
    assert distance(p, q) == pytest.approx(expected)
    assert distance(p * 4e307, q * 1e-300) == pytest.approx(expected)
    assert distance(p * 1e-300, q * 4e307) == pytest.approx(expected)
    np.testing.assert_array_equal(p, original_p)
    np.testing.assert_array_equal(q, original_q)


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize(
    ("p", "q"),
    [
        ([], []),
        (1, 1),
        ([1, 0], [1]),
        (np.ones((2, 3)), np.ones((3, 3))),
        (np.ones((2, 1)), np.ones((2, 3))),
        ([[1, 0], [0, 0]], [1, 0]),
        ([0, 0], [1, 0]),
        ([-1, 2], [1, 0]),
        ([np.nan, 1], [1, 0]),
        ([np.inf, 1], [1, 0]),
        (np.array([1 + 1j, 0]), [1, 0]),
    ],
)
def test_invalid_weights(distance: Distance, p: ArrayLike, q: ArrayLike) -> None:
    r"""Reject invalid inputs symmetrically instead of returning NaN distances."""
    with pytest.raises(ValueError, match="Weights must"):
        distance(p, q)
    with pytest.raises(ValueError, match="Weights must"):
        distance(q, p)


def test_small_distances() -> None:
    r"""Resolve tiny positive distances that the arccos formula rounds to zero."""
    eps = 1e-20
    p, q = [1, 0], [1, eps]
    assert fisher_rao_distance(p, q) == pytest.approx(2 * np.sqrt(eps), abs=0)
    assert hellinger_distance(p, q) == pytest.approx(np.sqrt(eps / 2), abs=0)
    assert total_variation_distance(p, q) > 0


def test_hellinger_fisher_rao_relation() -> None:
    r"""The two square-root geometries obey the documented chord–arc relation."""
    p, q = [3, 1, 1], [4, 2, 1]
    assert hellinger_distance(p, q) == pytest.approx(
        np.sqrt(2) * np.sin(fisher_rao_distance(p, q) / 4)
    )


@pytest.mark.parametrize("distance", DISTANCES)
def test_return_shape(distance: Distance) -> None:
    r"""Unbatched inputs return NumPy scalars; batches return float64 arrays."""
    scalar = distance([1, 0], [0, 1])
    batch = distance([[1, 0]], [0, 1])
    assert isinstance(scalar, np.float64)
    assert isinstance(batch, np.ndarray)
    assert scalar.shape == ()
    assert batch.shape == (1,)
    assert scalar.dtype == np.float64
    assert batch.dtype == np.float64


@pytest.mark.parametrize(
    ("distance", "diameter", "radius"),
    [
        (fisher_rao_distance, np.pi, 2 * np.arccos(0.5)),
        (hellinger_distance, 1, np.sqrt(0.5)),
        (total_variation_distance, 1, 0.75),
    ],
)
def test_batched_simplex_geometry(
    distance: Distance, diameter: float, radius: float
) -> None:
    r"""Broadcast vertices to all pairs and compare each vertex to the center."""
    vertices, center = np.eye(4), np.full(4, 0.25)
    pairwise = distance(vertices[:, None, :], vertices[None, :, :])
    assert pairwise.shape == (4, 4)
    np.testing.assert_allclose(pairwise, diameter * (1 - np.eye(4)))
    np.testing.assert_allclose(distance(vertices, center), np.full(4, radius))


@pytest.mark.parametrize("distance", DISTANCES)
def test_batched_normalization(distance: Distance) -> None:
    r"""Each distribution is normalized independently before batch broadcasting."""
    p = np.array([[3.0, 0.0, 1.0], [1.0, 2.0, 0.0]])
    q = np.array([[4.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
    expected = np.array([[distance(x, y) for y in q] for x in p])
    scaled_p = p[:, None, :] * np.array([1e-300, 4e307])[:, None, None]
    scaled_q = q[None, :, :] * np.array([4e307, 1e-300])[None, :, None]
    original_p, original_q = scaled_p.copy(), scaled_q.copy()
    result = distance(scaled_p, scaled_q)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(scaled_p, original_p)
    np.testing.assert_array_equal(scaled_q, original_q)


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("batch_shape", [(0,), (2, 0)])
def test_empty_batches(distance: Distance, batch_shape: tuple[int, ...]) -> None:
    r"""Empty batch axes are allowed when the category axis is nonempty."""
    batch = np.empty((*batch_shape, 3))
    result = distance(batch, [1, 0, 0])
    assert isinstance(result, np.ndarray)
    assert result.shape == batch_shape
    assert result.dtype == np.float64
