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


# Float[..., m, n] -> Float[...]
def erank(x: Tensor, /) -> Tensor:
    r"""Compute the effective rank of a matrix.

    $$\begin{aligned}
        pᵢ &≔ \frac{σᵢ}{∑ⱼ σⱼ} \\
        \op{erank}(A) &≔ e^{H(p)} = \exp\Bigl(-∑ᵢ pᵢ \log pᵢ\Bigr)
    \end{aligned}$$

    By definition, the effective rank is equal to the exponential of the entropy of the
    distribution of the singular values. Zero probabilities contribute zero to the
    entropy. A matrix whose singular values are all zero produces NaN.

    References:
        - `The effective rank: A measure of effective dimensionality \
          Olivier Roy, Martin Vetterli \
          `15th European Signal Processing Conference (EUSIPCO), 2007 \
          <https://ieeexplore.ieee.org/document/7098875>
    """
    σ = torch.linalg.svdvals(x)
    σ = σ / torch.linalg.norm(σ, ord=1, dim=-1)
    entropy = torch.special.entr(σ).sum(dim=-1)
    return torch.exp(entropy)


# Float[..., m, n] -> Float[...]
def relerank(x: Tensor, /) -> Tensor:
    r"""Compute the relative effective rank of a matrix.

    This is the effective rank divided by $\min(m,n)$ for an $m×n$ matrix.
    """
    return erank(x) / min(x.shape[-2:])


# Float[..., m, n] -> Float[...]
def col_corr(x: Tensor, /) -> Tensor:
    r"""Compute the mean absolute cosine similarity between distinct columns.

    $$ \frac{1}{n(n-1)}∑_{i≠j}\frac{|xᵢᵀxⱼ|}{‖xᵢ‖₂‖xⱼ‖₂} $$

    Here $xᵢ$ denotes column $i$ of a two-dimensional matrix. Columns are not
    centered. Zero columns or fewer than two columns can produce NaN.
    """
    _, n = x.shape[-2:]
    u = torch.linalg.norm(x, dim=0)
    xx = torch.einsum("...n, ...k -> ...nk", u, u)
    xtx = torch.einsum("...mn, ...mk  -> ...nk", x, x)
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    c = eye - xtx / xx
    return c.abs().sum(dim=(-2, -1)) / (n * (n - 1))


# Float[..., m, n] -> Float[...]
def row_corr(x: Tensor, /) -> Tensor:
    r"""Compute the mean absolute cosine similarity between distinct rows.

    $$ \frac{1}{m(m-1)}∑_{i≠j}\frac{|xᵢᵀxⱼ|}{‖xᵢ‖₂‖xⱼ‖₂} $$

    Here $xᵢ$ denotes row $i$ of a two-dimensional matrix. Rows are not
    centered. Zero rows or fewer than two rows can produce NaN.
    """
    m, _ = x.shape[-2:]
    v = torch.linalg.norm(x, dim=1)
    xx = torch.einsum("...m, ...k -> ...mk", v, v)
    xxt = torch.einsum("...mn, ...kn  -> ...mk", x, x)
    eye = torch.eye(m, dtype=x.dtype, device=x.device)
    c = eye - xxt / xx
    return c.abs().sum(dim=(-2, -1)) / (m * (m - 1))


# Float[..., n, n] -> Float[..., n, n]
def closest_symmetric(x: Tensor, /, *, dim: tuple[int, int] = (-2, -1)) -> Tensor:
    r"""Return the symmetric part of a square matrix along `dim`.

    $$ \frac{A+Aᵀ}{2} = \argmin_{X: Xᵀ = X} ‖A-X‖_F $$
    """
    rowdim, coldim = dim
    return (x + x.swapaxes(rowdim, coldim)) / 2


# Float[..., n, n] -> Float[..., n, n]
def closest_skew(x: Tensor, /, *, dim: tuple[int, int] = (-2, -1)) -> Tensor:
    r"""Return the skew-symmetric part of a square matrix along `dim`.

    $$ \frac{A-Aᵀ}{2} = \argmin_{X: Xᵀ = -X} ‖A-X‖_F $$
    """
    rowdim, coldim = dim
    return (x - x.swapaxes(rowdim, coldim)) / 2


# Float[..., n, n] -> Float[..., n, n]
def closest_orthogonal(x: Tensor, /) -> Tensor:
    r"""Return a closest orthogonal matrix to a real square matrix.

    $$ \argmin_{X: XᵀX = 𝕀} ‖A-X‖_F $$
    """
    U, _, Vt = torch.linalg.svd(x, full_matrices=True)
    Q = torch.einsum("...ij, ...jk->...ik", U, Vt)
    return Q


# Float[..., n, n] -> Float[..., n, n]
def closest_diagonal(x: Tensor, /) -> Tensor:
    r"""Return the diagonal part of a square matrix.

    $$ \argmin_{X: X⊙𝕀 = X} ‖A-X‖_F $$
    """
    d = torch.diagonal(x, dim1=-2, dim2=-1)
    return torch.diag_embed(d)


# Float[..., m, n], Float[..., m, n] -> Float[...]
def reldist(x: Tensor, y: Tensor, /) -> Tensor:
    r"""Compute the relative Frobenius distance between two matrices.

    $$ \frac{ ‖x-y‖_F }{ ‖y‖_F } $$

    Return zero when `y` has zero Frobenius norm, regardless of `x`.
    """
    r = torch.linalg.matrix_norm(x - y, ord="fro", dim=(-2, -1))
    yy = torch.linalg.matrix_norm(y, ord="fro", dim=(-2, -1))
    zero = torch.tensor(0.0, dtype=torch.float32, device=x.device)
    return torch.where(yy != 0, r / yy, zero)


# Float[..., n, n] -> Float[...]
def reldist_diagonal(x: Tensor, /) -> Tensor:
    r"""Compute the relative distance to being a diagonal matrix.

    $$ \frac{ ‖A-\diag(A)‖_F }{ ‖A‖_F } $$

    Return zero for a zero matrix.
    """
    return reldist(closest_diagonal(x), x)


# Float[..., n, n] -> Float[...]
def reldist_symmetric(x: Tensor, /) -> Tensor:
    r"""Compute the relative Frobenius distance to the closest symmetric matrix.

    Return zero for a zero matrix.
    """
    return reldist(closest_symmetric(x), x)


# Float[..., n, n] -> Float[...]
def reldist_skew(x: Tensor, /) -> Tensor:
    r"""Compute the relative Frobenius distance to the closest skew-symmetric matrix.

    Return zero for a zero matrix.
    """
    return reldist(closest_skew(x), x)


# Float[..., n, n] -> Float[...]
def reldist_orthogonal(x: Tensor, /) -> Tensor:
    r"""Compute the relative distance to the closest orthogonal matrix.

    $$ \min_{X: XᵀX = 𝕀} \frac{ ‖A-X‖_F }{ ‖A‖_F } $$

    Return zero for a zero matrix, following `reldist`.
    """
    return reldist(closest_orthogonal(x), x)


# Float[..., n, n] -> Float[...]
def stiffness_ratio(x: Tensor, /) -> Tensor:
    r"""Compute the stiffness ratio of a square matrix.

    $$ \frac{\maxᵢ |\Re(λᵢ)|}{\minᵢ |\Re(λᵢ)|} $$

    All eigenvalues must satisfy $\Re(λᵢ)<0$; otherwise return NaN.

    References:
        - J. D. Lambert \
          Numerical Methods for Ordinary Differential Systems: The Initial Value Problem \
          ISBN: 978-0-471-92990-1
    """
    x = x.to(dtype=torch.complex128)
    λ = torch.linalg.eigvals(x)
    λ = λ.real
    # Get smallest non-zero eigenvalue.
    maxvals = λ.amax(dim=-1)
    minvals = λ.amin(dim=-1)
    return torch.where(maxvals < 0, minvals / maxvals, torch.nan)


# Float[..., n, n] -> Float[...]
def spectral_radius(x: Tensor, /) -> Tensor:
    r"""Return the spectral radius $\maxᵢ |λᵢ|$."""
    λ = torch.linalg.eigvals(x)
    return λ.abs().amax(dim=-1)


# Float[..., n, n] -> Float[...]
def spectral_abscissa(x: Tensor, /) -> Tensor:
    r"""Return the spectral abscissa $\maxᵢ \Re(λᵢ)$."""
    λ = torch.linalg.eigvals(x)
    return λ.real.amax(dim=-1)


def _apply_keepdim(
    x: Tensor, /, *, dim: tuple[int, int], keepdim: bool = False
) -> Tensor:
    r"""Restore reduced dimensions as singleton axes when `keepdim=True`.

    The reduction must have been performed as follows:

    1. Move the axes in `dim`, in order, to the final dimensions.
    2. Reduce over those final dimensions to produce `x`.

    Return `x` unchanged when `keepdim=False`.
    """
    if not keepdim:
        return x

    rowdim, coldim = dim
    dims = (rowdim, coldim) if abs(rowdim) < abs(coldim) else (coldim, rowdim)
    for d in dims:
        x = x.unsqueeze(d)
    return x


def logarithmic_norm(
    x: Tensor,  # Float[..., n, n]
    /,
    *,
    p: float = 2.0,
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = False,
    scaled: bool = False,
) -> Tensor:  # Float[...]
    r"""Compute the logarithmic norm of a matrix.

    $$ μₚ(A) ≔ \lim_{ε→0⁺} \frac{‖𝕀+εA‖ₚ-1}{ε} $$

    For real matrices, the standard logarithmic norm has these special cases:

    - $p=+∞$: Maximum row sum, using absolute off-diagonal entries.
    - $p=2$: Maximum eigenvalue of the symmetric part of $A$.
    - $p=1$: Maximum column sum, using absolute off-diagonal entries.

    Diagonal entries contribute their real parts to the sums. Negative orders
    select the corresponding minima in this implementation. The current axis
    reductions exchange the row-sum and column-sum cases for $p=±1$ and $p=±∞$.
    Only $p=±1,±2,±∞$ are supported. The matrix must be square;
    `scaled=True` raises `NotImplementedError`.

    References:
        - [Logarithmic norm](https://en.wikipedia.org/wiki/Logarithmic_norm).
        - [What Is the Logarithmic Norm?](https://nhigham.com/2022/01/18/what-is-the-logarithmic-norm/).
        - The logarithmic norm. History and modern theory \
          Gustaf Söderlind, BIT Numerical Mathematics, 2006 \
          <https://link.springer.com/article/10.1007/s10543-006-0069-9>
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
    if p == torch.inf:
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return x.amax(dim=coldim - shift, keepdim=keepdim)
    if p == -torch.inf:
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return x.amin(dim=coldim - shift, keepdim=keepdim)

    raise NotImplementedError("Currently only p=±1,±2,±inf are supported.")


def schatten_norm(
    x: Tensor,  # Float[..., m, n]
    /,
    *,
    p: float = 2.0,
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = False,
    scaled: bool = False,
) -> Tensor:  # Float[...]
    r"""Compute a singular-value norm of order $p$.

    The standard Schatten norm is the vector norm of the singular values:

    $$ ‖A‖ₚ ≔ \Bigl(∑ᵢ σᵢᵖ\Bigr)^{1/p} $$

    | Order  | Standard interpretation                                                  |
    |--------|--------------------------------------------------------------------------|
    | $p=+∞$ | Maximum singular value (spectral norm).                                  |
    | $p=2$  | Frobenius norm (square root of the sum of squared singular values).      |
    | $p=1$  | Nuclear norm (sum of singular values).                                   |
    | $p=0$  | Number of nonzero singular values (rank).                                |
    | $p=-1$ | Reciprocal of the sum of reciprocal singular values.                     |
    | $p=-2$ | Reciprocal square root of the sum of reciprocal squared singular values. |
    | $p=-∞$ | Minimum singular value.                                                  |

    The implementation omits zero singular values. For finite nonzero `p`, it
    normalizes singular values by their maximum without restoring that scale.
    With `scaled=True`, it averages powers instead of summing them. For `p=0`,
    the scaled result is the geometric mean of nonzero singular values.
    For infinite orders, it returns the largest or smallest nonzero singular
    value; an all-zero matrix produces NaN in these cases.

    References:
        - [Matrix norms: Schatten norms](https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms).
        - [Schatten norms](https://en.wikipedia.org/wiki/Schatten_norms).
    """
    if not torch.is_floating_point(x):
        x = x.to(dtype=torch.float)

    rowdim, coldim = dim

    x = x.swapaxes(rowdim, -2).swapaxes(coldim, -1)
    σ = torch.linalg.svdvals(x)
    m = σ != 0

    if p == 0:
        if scaled:
            σ = torch.where(m, σ, torch.nan)
            result = geometric_mean(σ, dim=-1)
        else:
            result = m.sum(dim=-1)
        return _apply_keepdim(result, dim=dim, keepdim=keepdim)

    if p == torch.inf:
        σ = torch.where(m, σ, -torch.inf)
        maxvals = σ.amax(dim=-1)
        maxvals = torch.where(maxvals == -torch.inf, torch.nan, maxvals)
        return _apply_keepdim(maxvals, dim=dim, keepdim=keepdim)
    if p == -torch.inf:
        σ = torch.where(m, σ, torch.inf)
        minvals = σ.amin(dim=-1)
        minvals = torch.where(minvals == torch.inf, torch.nan, minvals)
        return _apply_keepdim(minvals, dim=dim, keepdim=keepdim)

    σ = torch.where(m, σ, -torch.inf)
    σ_max = σ.amax(dim=-1)
    σ = torch.where(m, σ, torch.nan)
    σ = σ / σ_max

    result = (
        σ.pow(p).nanmean(dim=-1).pow(1 / p)
        if scaled
        else σ.pow(p).nansum(dim=-1).pow(1 / p)
    )
    return _apply_keepdim(result, dim=dim, keepdim=keepdim)


def matrix_norm(
    x: Tensor,  # Float[..., m, n]
    /,
    *,
    dim: tuple[int, int] = (-2, -1),
    p: float = 2.0,
    q: float = 2.0,
    keepdim: bool = False,
    scaled: bool = False,
) -> Tensor:  # Float[...]
    r"""Compute the entry-wise matrix norm of orders $p$ and $q$.

    $$ ‖A‖_{p,q} ≔ \Bigl(∑ₙ \Bigl(∑ₘ |Aₘₙ|ᵖ\Bigr)^{q/p}\Bigr)^{1/q} $$

    Both orders default independently to two. The inner reduction uses `dim[0]`
    and order `p`; the outer reduction uses `dim[1]` and order `q`.
    For finite nonzero orders, the scaled version is defined as

    $$ ‖A‖_{p,q}^{\mathrm{scaled}} ≔ \Bigl(𝐄ₙ \Bigl(𝐄ₘ |Aₘₙ|ᵖ\Bigr)^{q/p}\Bigr)^{1/q} $$

    Here $𝐄$ is the arithmetic mean. Zero and infinite orders follow `tensor_norm`.

    References:
        - [Matrix norms: Entry-wise norms](https://en.wikipedia.org/wiki/Matrix_norm#L2,1_and_Lp,q_norms).
    """
    # convert to tuple
    dim = (dim[0] % x.ndim, dim[1] % x.ndim)  # absolufy dim
    # if keepdim[0] is `False`, then we need to adjust dim[1] accordingly:
    # this only happens if dim[0] < dim[1], otherwise dim[1] is already correct
    # 1 if dim[1] needs to change, 0 otherwise
    m = int(dim[0] < dim[1]) * (1 - int(keepdim))
    dim_inner, dim_outer = dim[0], dim[1] - m

    x = tensor_norm(x, p=p, dim=dim_inner, keepdim=keepdim, scaled=scaled)
    x = tensor_norm(x, p=q, dim=dim_outer, keepdim=keepdim, scaled=scaled)
    return x


def operator_norm(
    x: Tensor,  # Float[..., m, n]
    /,
    *,
    p: float = 2.0,
    dim: tuple[int, int] = (-2, -1),
    keepdim: bool = True,
    scaled: bool = False,
) -> Tensor:  # Float[...]
    r"""Compute an operator norm of order $p$ for a square matrix.

    For positive orders, the induced norm is defined by

    $$\begin{aligned}
        ‖x‖ₚ &≔ \Bigl(∑ₖ₌₁ⁿ |xₖ|ᵖ\Bigr)^{1/p} \\
        ‖A‖ₚ &≔ \sup_{x≠0} \frac{‖Ax‖ₚ}{‖x‖ₚ}
    \end{aligned}$$

    | Order  | Standard interpretation      |
    |--------|------------------------------|
    | $p=+∞$ | Maximum absolute row sum.    |
    | $p=2$  | Maximum singular value.      |
    | $p=1$  | Maximum absolute column sum. |
    | $p=-1$ | Minimum absolute column sum. |
    | $p=-2$ | Minimum singular value.      |
    | $p=-∞$ | Minimum absolute row sum.    |

    Negative orders select minima as an extension of the positive-order cases.
    Only $p=±1$, $±2$ and $±∞$ are supported. The current axis reductions exchange the
    row-sum and column-sum cases for $p=±1$ and $p=±∞$; some combinations of
    `dim` and `keepdim` can also select an invalid reduction axis.
    The current `scaled=True` factor uses axis indices rather than matrix sizes.

    References:
        - [Matrix norms induced by vector norms](https://en.wikipedia.org/wiki/Matrix_norm#Matrix_norms_induced_by_vector_norms).
        - [Operator norm](https://en.wikipedia.org/wiki/Operator_norm).
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
    if p == torch.inf:
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return c * x.amax(dim=coldim + shift, keepdim=keepdim)
    if p == -torch.inf:
        x = x.sum(dim=rowdim, keepdim=keepdim)
        return c * x.amin(dim=coldim + shift, keepdim=keepdim)

    raise NotImplementedError("Currently only p=±1,±2,±inf are supported.")
