r"""Test for checking how regular time series is."""

__all__ = [
    # Functions
    "approx_float_gcd",
    "coefficient_of_variation",
    "float_gcd",
    "geometric_std",
    "irregularity_coefficient",
    "is_quasiregular",
    "is_regular",
    "regularity_coefficient",
    "time_gcd",
]

import warnings
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from scipy import stats


def approx_float_gcd(
    x: ArrayLike, /, *, rtol: float = 1e-05, atol: float = 1e-08
) -> float:
    r"""Compute approximate GCD of multiple floats.

    .. math:: 𝗀𝖼𝖽_ϵ(x) = 𝗆𝖺𝗑\{y∣ ∀i : 𝖽𝗂𝗌𝗍(x_i, yℤ)≤ϵ\}

    .. warning:: This implementation does not work 100% correctly yet!

    References:
        - https://stackoverflow.com/q/45323619
    """
    warnings.warn(
        "The implementation of approx_float_gcd does not work 100% correctly yet!",
        RuntimeWarning,
        stacklevel=2,
    )
    x = np.abs(x).flatten()

    def _float_gcd(z: np.ndarray) -> float:
        n = len(z)
        t = np.min(z)
        if n == 1:
            return float(z[0])
        if n == 2:
            while np.abs(z[1]) > rtol * t + atol:
                z[0], z[1] = z[1], z[0] % z[1]
            return float(z[0])
        # n >= 3:
        out = np.empty(2)
        out[0] = _float_gcd(z[: (n // 2)])
        out[1] = _float_gcd(z[(n // 2) :])
        return _float_gcd(out)

    return _float_gcd(x)


def float_gcd(x: ArrayLike, /) -> float:
    r"""Compute the greatest common divisor (GCD) of a list of floats.

    Note:
        Since floats are rational numbers, this is a well-defined operation.
        We simply convert them to rational numbers and use the standard method.
    """
    x = np.asanyarray(x)

    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError("Input is not float!")

    mantissa_bits = {
        np.dtype("float16"): 11,
        np.dtype("float32"): 24,
        np.dtype("float64"): 53,
        np.dtype("float128"): 113,
    }[x.dtype]

    _, e = np.frexp(x)
    min_exponent = int(np.min(e))
    fac = mantissa_bits - min_exponent
    z = x * np.float_power(2, fac)  # <- use float_power to avoid overflow!

    if not np.allclose(z, np.rint(z)):
        raise ValueError("Numerical error accured during conversion!")

    gcd = np.gcd.reduce(np.rint(z).astype(int))
    gcd *= 2 ** (-fac)

    z = x / gcd
    z_int = np.rint(z).astype(int)

    if not np.allclose(z, z_int) or np.gcd.reduce(z_int) != 1:
        raise ValueError("Error check failed, computed GCD is not correct!")

    return cast("float", gcd)


def is_quasiregular(s: Series | DataFrame, /) -> bool:
    r"""Test if time series is quasi-regular.

    By definition, this is the case if all timedeltas are
    integer multiples of the minimal, non-zero timedelta of the series.
    """
    if isinstance(s, DataFrame):
        return is_quasiregular(Series(s.index))

    Δt = np.diff(s)
    zero = np.array(0, dtype=Δt.dtype)
    Δt_min = np.min(Δt[Δt > zero])
    z = Δt / Δt_min
    return np.allclose(z, np.rint(z))


def is_regular(s: Series | DataFrame) -> bool:
    r"""Test if time series is regular, i.e. iff $Δt_i$ is constant."""
    if isinstance(s, DataFrame):
        return is_regular(Series(s.index))

    Δt = np.diff(s)
    return bool(np.all(Δt == np.min(Δt)))


def regularity_coefficient(
    s: Series | DataFrame, /, *, ignore_duplicates: bool = True
) -> float:
    r"""Compute the regularity coefficient of a time series.

    The regularity coefficient is equal to the ratio of length of the smallest regular time-series
    that contains s and the length of s.

    .. math:: κ(𝐭) = \frac{(t_\max-t_\min)/𝗀𝖼𝖽(𝐭)}{|𝐭|}

    In particular, if the time-series is regular, $κ=1$, and if it is irregular, $κ=∞$.
    To make the time-series regular, one would have to insert additional
    :math:`(κ(𝐭)-1) | 𝐭 |`-many data-points.
    """
    if isinstance(s, DataFrame):
        return regularity_coefficient(Series(s.index))

    gcd = time_gcd(s)
    Δt = np.diff(s)
    if ignore_duplicates:
        zero = np.array(0, dtype=Δt.dtype)
        Δt = Δt[Δt > zero]
    coef: float = ((np.max(s) - np.min(s)) / gcd) / len(Δt)
    return coef


def time_gcd(s: Series, /) -> float:
    r"""Compute the greatest common divisor of datetime64/int/float data."""
    Δt = np.diff(s)
    zero = np.array(0, dtype=Δt.dtype)
    Δt = Δt[Δt > zero]

    if pd.api.types.is_timedelta64_dtype(Δt):
        Δt = Δt.astype("timedelta64[ns]").astype(int)
        gcd = np.gcd.reduce(Δt)
        return gcd.astype("timedelta64[ns]")
    if pd.api.types.is_integer_dtype(Δt):
        return np.gcd.reduce(Δt)
    if pd.api.types.is_float_dtype(Δt):
        return float_gcd(Δt)

    raise NotImplementedError(f"Data type {Δt.dtype=} not understood")


def irregularity_coefficient(s: ArrayLike, /, *, drop_zero: bool = True) -> float:
    r"""Compute the irregularity coefficient of a time differences.

    Args:
        s: Sequence of time stamps
        drop_zero: Whether to drop zero time differences (default: True)

    Returns:
        γ(T) = \max(∆T) / \gcd(∆T)
    """
    t = Series(np.asarray(s))
    dt: Series = t.iloc[1:] - t.iloc[:-1]

    if drop_zero:
        # NOTE: use equality instead of inequality to serve nulls
        mask = (dt == 0).fillna(value=False)
        dt = dt[~mask]

    # special case floating point numbers
    if pd.api.types.is_float_dtype(dt):
        # convert to a numpy float
        dt_float = dt.astype(np.float64)
        return float(dt_float.max() / float_gcd(dt_float))

    # special case integer numbers
    if pd.api.types.is_integer_dtype(dt):
        dt_int = dt
    # otherwise convert to integer
    else:
        try:
            dt_int = dt.astype("int64[pyarrow]")
        except Exception as e:
            e.add_note("Could not convert time differences to int64[pyarrow]")
            raise

    return float(np.max(dt_int) / np.gcd.reduce(dt_int))


def coefficient_of_variation(s: ArrayLike, /, *, drop_zero: bool = True) -> float:
    r"""Compute the coefficient of variation of a time differences.

    Args:
        s: Sequence of time stamps
        drop_zero: Whether to drop zero time differences (default: True)

    Returns:
        γ(T) = σ(∆T) / μ(∆T)
    """
    t = Series(np.asarray(s))
    dt: Series = t.iloc[1:] - t.iloc[:-1]

    if drop_zero:
        # NOTE: use equality instead of inequality to serve nulls
        mask = (dt == 0).fillna(value=False)
        dt = dt[~mask]

    return float(stats.variation(dt))


def geometric_std(s: ArrayLike, /, *, drop_zero: bool = True) -> float:
    r"""Compute the geometric standard deviation of a time differences.

    Args:
        s: Sequence of time stamps
        drop_zero: Whether to drop zero time differences (default: True)

    Returns:
        σ_g(T) = exp(σ(log(∆T)))
    """
    t: Series = Series(np.asarray(s))
    dt: Series = t.iloc[1:] - t.iloc[:-1]

    if drop_zero:
        # NOTE: use equality instead of inequality to serve nulls
        mask = (dt == 0).fillna(value=False)
        dt = dt[~mask]

    return float(stats.gstd(dt))
