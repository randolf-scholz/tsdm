r"""#TODO add module summary line.

#TODO add module description.
"""


import torch
from torch import Tensor, jit


@jit.script
def erank(x: Tensor) -> Tensor:
    r"""Compute the effective rank of a matrix.

    .. Signature:: ``(..., m, n) -> ...``

    By definition, the effective rank is equal to the exponential of the entropy of the
    distribution of the singular values.

    .. math:: \operatorname{erank}(A) = e^{H(\tfrac{𝛔}{‖𝛔‖_1})} = ∏ σ_{i}^{-σ_i}

    References
    ----------
    - | `The effective rank: A measure of effective dimensionality
        <https://ieeexplore.ieee.org/document/7098875>`_
      | Olivier Roy, Martin Vetterli
      | `15th European Signal Processing Conference (EUSIPCO), 2007
        <https://ieeexplore.ieee.org/xpl/conhome/7067185/proceeding>`_
    """
    σ = torch.linalg.svdvals(x)
    σ = σ / torch.linalg.norm(σ, ord=1, dim=-1)
    entropy = torch.special.entr(σ).sum(dim=-1)
    return torch.exp(entropy)


@jit.script
def col_corr(x: Tensor) -> Tensor:
    r"""Compute average column-wise correlation of a matrix.

    .. Signature:: ``(..., m, n) -> ...``
    """
    _, n = x.shape[-2:]
    u = torch.linalg.norm(x, dim=0)
    xx = torch.einsum("...i, ...j -> ...ij", u, u)
    xtx = torch.einsum("...ik, ...il  -> ...kl", x, x)
    I = torch.eye(n, dtype=x.dtype, device=x.device)
    c = I - xtx / xx
    return c.abs().sum(dim=(-2, -1)) / (n * (n - 1))


@jit.script
def row_corr(x: Tensor) -> Tensor:
    r"""Compute average column-wise correlation of a matrix.

    .. Signature:: ``(..., m, n) -> ...``
    """
    m, _ = x.shape[-2:]
    v = torch.linalg.norm(x, dim=1)
    xx = torch.einsum("...i, ...j -> ...ij", v, v)
    xxt = torch.einsum("...kj, ...lj  -> ...kl", x, x)
    I = torch.eye(m, dtype=x.dtype, device=x.device)
    c = I - xxt / xx
    return c.abs().sum(dim=(-2, -1)) / (m * (m - 1))


@jit.script
def closest_symm(x: Tensor) -> Tensor:
    r"""Symmetric part of square matrix.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \argmin_{X: X^⊤ = -X} ‖A-X‖
    """
    return (x + x.swapaxes(-1, -2)) / 2


@jit.script
def closest_skew(x: Tensor) -> Tensor:
    r"""Skew-Symmetric part of a matrix.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \argmin_{X: X^⊤ = X} ‖A-X‖
    """
    return (x - x.swapaxes(-1, -2)) / 2


@jit.script
def closest_orth(x: Tensor) -> Tensor:
    r"""Orthogonal part of a square matrix.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \argmin_{X: XᵀX = 𝕀} ‖A-X‖
    """
    U, _, Vt = torch.linalg.svd(x, full_matrices=True)
    Q = torch.einsum("...ij, ...jk->...ik", U, Vt)
    return Q


@jit.script
def closest_diag(x: Tensor) -> Tensor:
    r"""Diagonal part of a square matrix.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \argmin_{X: X⊙𝕀 = X} ‖A-X‖
    """
    d = torch.diagonal(x, dim1=-2, dim2=-1)
    return torch.diag_embed(d)


@jit.script
def matrix_reldist(x: Tensor, y: Tensor) -> Tensor:
    r"""Relative distance between two matrices.

    .. Signature:: ``[(..., m, n), (..., m, n)]  -> (..., n, n)``

    .. math::  ‖x-y‖/‖y‖
    """
    r = torch.linalg.matrix_norm(x - y, ord="fro", dim=(-2, -1))
    yy = torch.linalg.matrix_norm(y, ord="fro", dim=(-2, -1))
    zero = torch.tensor(0.0, dtype=torch.float32, device=x.device)
    return torch.where(yy != 0, r / yy, zero)


@jit.script
def reldist_diag(x: Tensor) -> Tensor:
    r"""Compute the relative distance to being a diagonal matrix.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: ‖A-X‖/‖A‖  X = \argmin_{X: X⊙𝕀 = X} ‖A-X‖
    """
    return matrix_reldist(closest_diag(x), x)


@jit.script
def reldist_symm(x: Tensor) -> Tensor:
    r"""Relative magnitude of closest_symm part.

    .. Signature:: ``(..., n, n) -> ...``
    """
    return matrix_reldist(closest_symm(x), x)


@jit.script
def reldist_skew(x: Tensor) -> Tensor:
    r"""Relative magnitude of skew-closest_symm part.

    .. Signature:: ``(..., n, n) -> ...``
    """
    return matrix_reldist(closest_skew(x), x)


@jit.script
def reldist_orth(x: Tensor) -> Tensor:
    r"""Relative magnitude of orthogonal part.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: \min_{Q: Q^⊤Q = 𝕀} ‖A-Q‖/‖A‖
    """
    return matrix_reldist(closest_orth(x), x)
