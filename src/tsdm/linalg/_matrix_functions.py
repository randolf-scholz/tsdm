r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Functions
    "closest_diag",
    "closest_orth",
    "closest_skew",
    "closest_symm",
    "col_corr",
    "erank",
    "relerank",
    "reldist",
    "reldist_diag",
    "reldist_orth",
    "reldist_skew",
    "reldist_symm",
    "row_corr",
    "stiffness_ratio",
    "spectral_radius",
    "spectral_abscissa",
    "logarithmic_norm",
]


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
def relerank(x: Tensor) -> Tensor:
    r"""Compute the relative effective rank of a matrix.

    .. Signature:: ``(..., m, n) -> ...``

    This is the effective rank scaled by $\min(m,n)$.
    """
    return erank(x) / min(x.shape[-2:])


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
def reldist(x: Tensor, y: Tensor) -> Tensor:
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
    return reldist(closest_diag(x), x)


@jit.script
def reldist_symm(x: Tensor) -> Tensor:
    r"""Relative magnitude of closest_symm part.

    .. Signature:: ``(..., n, n) -> ...``
    """
    return reldist(closest_symm(x), x)


@jit.script
def reldist_skew(x: Tensor) -> Tensor:
    r"""Relative magnitude of skew-closest_symm part.

    .. Signature:: ``(..., n, n) -> ...``
    """
    return reldist(closest_skew(x), x)


@jit.script
def reldist_orth(x: Tensor) -> Tensor:
    r"""Relative magnitude of orthogonal part.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: \min_{Q: Q^⊤Q = 𝕀} ‖A-Q‖/‖A‖
    """
    return reldist(closest_orth(x), x)


@jit.script
def stiffness_ratio(x: Tensor) -> Tensor:
    r"""Compute the stiffness ratio of a matrix.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: \frac{|\Re(λ_\max)|}{|\Re{λ_\min}|}

    Only applicable if $\Re(λ_i)<0$ for all $i$.

    References
    ----------
    - | Numerical Methods for Ordinary Differential Systems: The Initial Value Problem
      | J. D. Lambert
      | ISBN: 978-0-471-92990-1
    """
    x = x.to(dtype=torch.complex128)
    λ = torch.linalg.eigvals(x)
    λ = λ.real
    # Get smallest non-zero eigenvalue.
    maxvals = λ.amax(dim=-1)
    minvals = λ.amin(dim=-1)
    return torch.where(maxvals < 0, minvals / maxvals, float("nan"))


@jit.script
def spectral_radius(x: Tensor) -> Tensor:
    r"""Return $\max_i | λ_i |$.

    .. Signature:: ``(..., n, n) -> ...``
    """  # noqa: RST219
    λ = torch.linalg.eigvals(x)
    return λ.abs().amax(dim=-1)


@jit.script
def spectral_abscissa(x: Tensor) -> Tensor:
    r"""Return $\max_i \Re(λ_i)$.

    .. Signature:: ``(..., n, n) -> ...``
    """
    λ = torch.linalg.eigvals(x)
    return λ.real.amax(dim=-1)


@jit.script
def logarithmic_norm(x: Tensor, p: float = 2.0) -> Tensor:
    r"""Compute the logarithmic norm of a matrix.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: \lim_{ε→0⁺} \frac{‖𝕀+εA‖_p-1}{ε}

    References
    ----------
    - `What Is the Logarithmic Norm? <https://nhigham.com/2022/01/18/what-is-the-logarithmic-norm/>_`
    - | The logarithmic norm. History and modern theory
      | Gustaf Söderlind, BIT Numerical Mathematics, 2006
      | <https://link.springer.com/article/10.1007/s10543-006-0069-9>_
    """
    if p == 2:
        return spectral_abscissa(closest_symm(x))

    m = torch.eye(x.shape[-1], dtype=torch.bool)
    x = torch.where(m, x.real, x.abs())

    if p == 1:
        return x.sum(dim=-1).amax(dim=-1)
    if p == float("inf"):
        return x.sum(dim=-2).amax(dim=-1)

    raise NotImplementedError("Currently only p=1,2,inf are supported.")
