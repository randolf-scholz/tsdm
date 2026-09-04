r"""Functions specific to rank-2 tensors."""

__all__ = [
    # Functions
    "closest_diagonal",
    "closest_orthogonal",
    "closest_skew",
    "closest_symmetric",
    "col_corr",
    "erank",
    "logarithmic_norm",
    "matrix_norm",
    "operator_norm",
    "reldist",
    "reldist_diagonal",
    "reldist_orthogonal",
    "reldist_skew",
    "reldist_symmetric",
    "relerank",
    "row_corr",
    "schatten_norm",
    "spectral_abscissa",
    "spectral_radius",
    "stiffness_ratio",
]

import torch
from torch import Tensor

from .tensor_functions import geometric_mean, tensor_norm


def erank(x: Tensor, /) -> Tensor:
    r"""Compute the effective rank of a matrix.

    .. signature:: ``(..., m, n) -> ...``

    .. math:: \operatorname{erank}(A) ≔ e^{H(\frac{𝛔}{‖𝛔‖_1})}
        = ∏ \bigl(\frac{σ_i}{‖σ_i‖}\bigr)^{- \frac{σ_i}{‖σ_i‖}}

    By definition, the effective rank is equal to the exponential of the entropy of the
    distribution of the singular values.

    References:
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


def relerank(x: Tensor, /) -> Tensor:
    r"""Compute the relative effective rank of a matrix.

    .. signature:: ``(..., m, n) -> ...``

    This is the effective rank scaled by $\min(m,n)$.
    """
    return erank(x) / min(x.shape[-2:])


def col_corr(x: Tensor, /) -> Tensor:
    r"""Compute average column-wise correlation of a matrix.

    .. signature:: ``(..., m, n) -> ...``

    .. math:: \frac{1}{n(n-1)}\left\|𝕀ₙ - \frac{XᵀX}{\diag(XᵀX)⊗\diag(XᵀX)}\right\|_{1,1}
    """
    _, n = x.shape[-2:]
    u = torch.linalg.norm(x, dim=0)
    xx = torch.einsum("...n, ...k -> ...nk", u, u)
    xtx = torch.einsum("...mn, ...mk  -> ...nk", x, x)
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    c = eye - xtx / xx
    return c.abs().sum(dim=(-2, -1)) / (n * (n - 1))


def row_corr(x: Tensor, /) -> Tensor:
    r"""Compute average column-wise correlation of a matrix.

    .. signature:: ``(..., m, n) -> ...``

    .. math:: \frac{1}{m(m-1)}\left\|𝕀ₘ - \frac{XXᵀ}{diag(XXᵀ)⊗diag(XXᵀ)}\right\|_{1,1}
    """
    m, _ = x.shape[-2:]
    v = torch.linalg.norm(x, dim=1)
    xx = torch.einsum("...m, ...k -> ...mk", v, v)
    xxt = torch.einsum("...mn, ...kn  -> ...mk", x, x)
    eye = torch.eye(m, dtype=x.dtype, device=x.device)
    c = eye - xxt / xx
    return c.abs().sum(dim=(-2, -1)) / (m * (m - 1))


def closest_symmetric(x: Tensor, /, *, dim: tuple[int, int] = (-2, -1)) -> Tensor:
    r"""Symmetric part of square matrix.

    .. signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \argmin_{X: X^⊤ = -X} ‖A-X‖
    """
    rowdim, coldim = dim
    return (x + x.swapaxes(rowdim, coldim)) / 2


def closest_skew(x: Tensor, /, *, dim: tuple[int, int] = (-2, -1)) -> Tensor:
    r"""Skew-Symmetric part of a matrix.

    .. signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \argmin_{X: X^⊤ = X} ‖A-X‖
    """
    rowdim, coldim = dim
    return (x - x.swapaxes(rowdim, coldim)) / 2


def closest_orthogonal(x: Tensor, /) -> Tensor:
    r"""Orthogonal part of a square matrix.

    .. signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \argmin_{X: XᵀX = 𝕀} ‖A-X‖
    """
    U, _, Vt = torch.linalg.svd(x, full_matrices=True)
    Q = torch.einsum("...ij, ...jk->...ik", U, Vt)
    return Q


def closest_diagonal(x: Tensor, /) -> Tensor:
    r"""Diagonal part of a square matrix.

    .. math:: \argmin_{X: X⊙𝕀 = X} ‖A-X‖

    .. signature:: ``(..., n, n) -> (..., n, n)``
    """
    d = torch.diagonal(x, dim1=-2, dim2=-1)
    return torch.diag_embed(d)


def reldist(x: Tensor, y: Tensor, /) -> Tensor:
    r"""Relative distance between two matrices.

    .. signature:: ``[(..., m, n), (..., m, n)]  -> (..., n, n)``

    .. math::  \frac{‖x-y‖}{‖y‖}
    """
    r = torch.linalg.matrix_norm(x - y, ord="fro", dim=(-2, -1))
    yy = torch.linalg.matrix_norm(y, ord="fro", dim=(-2, -1))
    zero = torch.tensor(0.0, dtype=torch.float32, device=x.device)
    return torch.where(yy != 0, r / yy, zero)


def reldist_diagonal(x: Tensor, /) -> Tensor:
    r"""Compute the relative distance to being a diagonal matrix.

    .. signature:: ``(..., n, n) -> ...``

    .. math:: \frac{‖A-X‖}{‖A‖}  X = \argmin_{X: X⊙𝕀 = X} ‖A-X‖
    """
    return reldist(closest_diagonal(x), x)


def reldist_symmetric(x: Tensor, /) -> Tensor:
    r"""Relative magnitude of closest_symm part.

    .. signature:: ``(..., n, n) -> ...``
    """
    return reldist(closest_symmetric(x), x)


def reldist_skew(x: Tensor, /) -> Tensor:
    r"""Relative magnitude of skew-closest_symm part.

    .. signature:: ``(..., n, n) -> ...``
    """
    return reldist(closest_skew(x), x)


def reldist_orthogonal(x: Tensor, /) -> Tensor:
    r"""Relative magnitude of orthogonal part.

    .. signature:: ``(..., n, n) -> ...``

    .. math:: \min_{X: X^⊤X = 𝕀} \frac{‖A-X‖}{‖A‖}
    """
    return reldist(closest_orthogonal(x), x)


def stiffness_ratio(x: Tensor, /) -> Tensor:
    r"""Compute the stiffness ratio of a matrix.

    .. signature:: ``(..., n, n) -> ...``

    .. math:: \frac{|\Re(λ_\max)|}{|\Re(λ_\min)|}

    Only applicable if $\Re(λ_i)<0$ for all $i$.

    References:
        - | Numerical Methods for Ordinary Differential Systems: The Initial Value Problem
          | J. D. Lambert, ISBN: 978-0-471-92990-1
    """
    x = x.to(dtype=torch.complex128)
    λ = torch.linalg.eigvals(x)
    λ = λ.real
    # Get smallest non-zero eigenvalue.
    maxvals = λ.amax(dim=-1)
    minvals = λ.amin(dim=-1)
    return torch.where(maxvals < 0, minvals / maxvals, float("nan"))


def spectral_radius(x: Tensor, /) -> Tensor:
    r"""Return $\max_i | λ_i | $.

    .. signature:: ``(..., n, n) -> ...``
    """
    λ = torch.linalg.eigvals(x)
    return λ.abs().amax(dim=-1)


def spectral_abscissa(x: Tensor, /) -> Tensor:
    r"""Return $\max_i \Re(λ_i)$.

    .. signature:: ``(..., n, n) -> ...``
    """
    λ = torch.linalg.eigvals(x)
    return λ.real.amax(dim=-1)


def _apply_keepdim(
    x: Tensor, /, *, dim: tuple[int, int], keepdim: bool = False
) -> Tensor:
    r"""Insert dimensions in the right places.

    We assume x was some tensor to which a reduction was applied, such that

    1. The affected axes were mapped, in order, to the last dimensions of x.
    2. The reduction was performed over the last dimensions of x.
    3. We now want to insert the dimensions back into x at the right places.
    """
    if not keepdim:
        return x

    rowdim, coldim = dim
    dims = (rowdim, coldim) if abs(rowdim) < abs(coldim) else (coldim, rowdim)
    for d in dims:
        x = x.unsqueeze(d)
    return x


def logarithmic_norm(
    x: Tensor,
    /,
    *,
    p: float = 2.0,
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = False,
    scaled: bool = False,
) -> Tensor:
    r"""Compute the logarithmic norm of a matrix.

    .. math:: \lim_{ε→0⁺} \frac{‖𝕀+εA‖ₚ-1}{ε}

    .. signature:: ``(..., n, n) -> ...``

    Special cases:

    - p=+∞: maxiumum rowsum, using real value for diagonal
    - p=+2: maximum eigenvalue of symmetric part of A
    - p=+1: maximum columnsum, using real value for diagonal
    - p=-1: minimum columnsum, using real value for diagonal
    - p=-2: minimum eigenvalue of symmetric part of A
    - p=-∞: minimum rowsum, using real value for diagonal

    References:
        - Logarithmic Norm https://en.wikipedia.org/wiki/Logarithmic_norm
        - `What Is the Logarithmic Norm? <https://nhigham.com/2022/01/18/what-is-the-logarithmic-norm/>_`
        - | The logarithmic norm. History and modern theory
          | Gustaf Söderlind, BIT Numerical Mathematics, 2006
          | https://link.springer.com/article/10.1007/s10543-006-0069-9
    """
    if scaled:
        raise NotImplementedError("Implementation not validated!")

    rowdim, coldim = dim
    rowdim = rowdim % x.ndim
    coldim = coldim % x.ndim
    dim = (rowdim, coldim)
    M, N = x.shape[rowdim], x.shape[coldim]

    if M != N:
        raise ValueError("Matrix must be square.")

    if p == 2:
        x = closest_symmetric(x, dim=dim)
        x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
        λ = torch.linalg.eigvals(x)
        r = λ.real.amax(dim=-1)
        return _apply_keepdim(r, dim=dim, keepdim=keepdim)
    if p == -2:
        x = closest_symmetric(x, dim=dim)
        x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
        λ = torch.linalg.eigvals(x)
        r = λ.real.amin(dim=-1)
        return _apply_keepdim(r, dim=dim, keepdim=keepdim)

    m = torch.eye(N, dtype=torch.bool, device=x.device)
    x = torch.where(m, x.real, x.abs())

    shift = int(coldim < rowdim) * (1 - int(keepdim))
    if p == 1:
        x = x.sum(dim=coldim, keepdim=keepdim)
        return x.amax(dim=rowdim - shift, keepdim=keepdim)
    if p == -1:
        x = x.sum(dim=coldim, keepdim=keepdim)
        return x.amin(dim=rowdim - shift, keepdim=keepdim)
    shift = int(rowdim < coldim) * (1 - int(keepdim))
    if p == float("inf"):
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return x.amax(dim=coldim - shift, keepdim=keepdim)
    if p == -float("inf"):
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return x.amin(dim=coldim - shift, keepdim=keepdim)

    raise NotImplementedError("Currently only p=±1,±2,±inf are supported.")


def schatten_norm(
    x: Tensor,
    /,
    *,
    p: float = 2.0,
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = False,
    scaled: bool = False,
) -> Tensor:
    r"""Schatten norm $p$-th order.

    .. signature:: ``(..., m, n) -> ...``

    .. math::  ‖A‖_p^p ≔ \tr(|A|^p) = ∑_i σ_i^p

    The Schatten norm is equivalent to the vector norm of the singular values.

    - $p=+∞$: Maximum Singular Value, equivalent to spectral norm $‖A‖_2$.
    - $p=2$: Frobius Norm (sum of squared values)
    - $p=1$: Nuclear Norm (sum of singular values)
    - $p=0$: Number of non-zero singular values. Equivalent to rank.
    - $p=-1$: Reciprocal sum of singular values.
    - $p=-2$: Reciprocal sum of squared singular values.
    - $p=+∞$: Minimal Singular Value

    References:
        - Matrix Norms (schatten): https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms
        - Schatten Norms: https://en.wikipedia.org/wiki/Schatten_norms
    """
    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float)

    rowdim, coldim = dim

    x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
    σ = torch.linalg.svdvals(x)
    m = σ != 0

    if p == 0:
        if scaled:
            σ = torch.where(m, σ, float("nan"))
            result = geometric_mean(σ, axis=-1)
        else:
            result = m.sum(dim=-1)
        return _apply_keepdim(result, dim=dim, keepdim=keepdim)

    if p == float("+inf"):
        σ = torch.where(m, σ, float("-inf"))
        maxvals = σ.amax(dim=-1)
        maxvals = torch.where(maxvals == float("-inf"), float("nan"), maxvals)
        return _apply_keepdim(maxvals, dim=dim, keepdim=keepdim)
    if p == float("-inf"):
        σ = torch.where(m, σ, float("+inf"))
        minvals = σ.amin(dim=-1)
        minvals = torch.where(minvals == float("+inf"), float("nan"), minvals)
        return _apply_keepdim(minvals, dim=dim, keepdim=keepdim)

    σ = torch.where(m, σ, float("-inf"))
    σ_max = σ.amax(dim=-1)
    σ = torch.where(m, σ, float("+nan"))
    σ = σ / σ_max

    result = (
        σ.pow(p).nanmean(dim=-1).pow(1 / p)
        if scaled
        else σ.pow(p).nansum(dim=-1).pow(1 / p)
    )
    return _apply_keepdim(result, dim=dim, keepdim=keepdim)


def matrix_norm(
    x: Tensor,
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    p: float = 2.0,
    q: float = 2.0,
    keepdim: bool = False,
    scaled: bool = False,
) -> Tensor:
    r"""Entry-Wise Matrix norm of $p,q$-th order.

    .. signature:: ``(..., m, n) -> ...``

    .. math:: ‖A‖_{p,q} ≔ \Bigl(∑_n \Bigl(∑_m |A_{mn}|^p\Bigr)^{q/p} \Bigr)^{1/q}

    If $q$ is not specified, then $q=p$ is used. The scaled version is defined as

    .. math:: ‖A‖_{p,q}^* ≔ \Bigl(𝐄_n \Bigl(𝐄_m |A_{mn}|^p\Bigr)^{q/p}\Bigr)^{1/q}

    where $𝐄$ is the averaging operator, which estimates the expected value $𝔼$.

    References:
        - Matrix Norms (entry-wise): https://en.wikipedia.org/wiki/Matrix_norm#L2,1_and_Lp,q_norms
    """
    # convert to tuple
    dim = (dim[0] % x.ndim, dim[1] % x.ndim)  # absolufy dim
    # if keepdim[0] is `False`, then we need to adjust dim[1] accordingly:
    # this only happens if dim[0] < dim[1], otherwise dim[1] is already correct
    # 1 if dim[1] needs to change, 0 otherwise
    m = int(dim[0] < dim[1]) * (1 - int(keepdim))
    dim_inner, dim_outer = dim[0], dim[1] - m

    x = tensor_norm(x, p=p, axis=dim_inner, keepdim=keepdim, scaled=scaled)
    x = tensor_norm(x, p=q, axis=dim_outer, keepdim=keepdim, scaled=scaled)
    return x


def operator_norm(
    x: Tensor,
    /,
    *,
    p: float = 2.0,
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = True,
    scaled: bool = False,
) -> Tensor:
    r"""Operator norm of $p$-th order.

    .. signature:: ``(..., m, n) -> ...``

    .. math::
        ‖x‖ₚ &≔ (   ∑_{k=0}^n |xₖ|ᵖ)^{1/p}  \text{scaled=False}               \\
        ‖x‖ₚ &≔ (⅟ₙ ∑_{k=0}^n |xₖ|ᵖ)^{1/p}  \text{scaled=True}                \\
        ‖A‖ₚ &≔ \sup_{x≠0} \frac{‖Ax‖ₚ}{‖x‖ₚ}

    +--------+--------------------------+---------+
    |        | regular                  | scaled  |
    +========+==========================+=========+
    | $p=+∞$ | maximum absolute row sum | ᵖ√(n/m) |
    +--------+--------------------------+---------+
    | $p=+2$ | maximum singular value   | ᵖ√(n/m) |
    +--------+--------------------------+---------+
    | $p=+1$ | maximum absolute col sum | ᵖ√(n/m) |
    +--------+--------------------------+---------+
    | $p=±0$ | ∞                        | ᵖ√(n/m) |
    +--------+--------------------------+---------+
    | $p=-1$ | minimum absolute col sum | ᵖ√(n/m) |
    +--------+--------------------------+---------+
    | $p=-2$ | minimum singular value   | ᵖ√(n/m) |
    +--------+--------------------------+---------+
    | $p=-∞$ | minimum absolute row sum | ᵖ√(n/m) |
    +--------+--------------------------+---------+

    References:
        - Matrix Norms (induced): https://en.wikipedia.org/wiki/Matrix_norm#Matrix_norms_induced_by_vector_norms
        - Operator Norm: https://en.wikipedia.org/wiki/Operator_norm
    """
    rowdim, coldim = dim
    if x.shape[rowdim] != x.shape[coldim]:
        raise ValueError("Matrix must be square.")

    # branchless version of c = (coldim / rowdim) ** (1 / p) if scaled else 1.0
    c = (coldim / rowdim) ** (float(scaled) / p)

    if p == 2:
        x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
        σ = torch.linalg.svdvals(x)
        r = σ.amax(dim=-1)
        return c * _apply_keepdim(r, dim=dim, keepdim=keepdim)
    if p == -2:
        x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
        σ = torch.linalg.svdvals(x)
        r = σ.amin(dim=-1)
        return c * _apply_keepdim(r, dim=dim, keepdim=keepdim)

    x = x.abs()
    shift = int(coldim < rowdim) * int(keepdim)

    if p == 1:
        x = x.sum(dim=coldim, keepdim=keepdim)
        return c * x.amax(dim=rowdim - shift, keepdim=keepdim)
    if p == -1:
        x = x.sum(dim=coldim, keepdim=keepdim)
        return c * x.amin(dim=rowdim - shift, keepdim=keepdim)
    if p == float("inf"):
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return c * x.amax(dim=coldim + shift, keepdim=keepdim)
    if p == -float("inf"):
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return c * x.amin(dim=coldim + shift, keepdim=keepdim)

    raise NotImplementedError("Currently only p=±1,±2,±inf are supported.")
