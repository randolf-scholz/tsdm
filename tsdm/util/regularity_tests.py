r"""
Test for checking how regular time series is.
"""

from numpy.typing import ArrayLike
from pandas import Series
import numpy as np
import numba


def float_gcd(x: ArrayLike) -> float:
    r"""Computes the greatest common divisor (GCD) off a list of floats.

    Note that since floats are rational numbers, this is well-defined.

    Parameters
    ----------
    x: ArrayLike

    Returns
    -------
    float:
        The GCD of the list
    """
    x = np.asanyarray(x)

    assert np.issubdtype(x.dtype, np.floating), "input is not float!"

    mantissa_bits = {
        np.float16:  11,
        np.float32:  24,
        np.float64:  53,
        np.float128: 113,
    }[x.dtype]

    m, e = np.frexp(x)
    min_exponent = int(np.min(e))
    fac = mantissa_bits - min_exponent

    z = x * 2 ** fac
    assert np.allclose(z, np.rint(z)), "something went wrong"

    gcd = np.gcd.reduce(np.rint(z).astype(int))
    gcd = gcd * 2**(-fac)

    z = x/gcd
    z_int = np.rint(z).astype(int)
    assert np.allclose(z, z_int), "Not a GCD!"
    assert np.gcd.reduce(z_int) == 1, "something went wrong"
    return gcd


def approx_float_gcd(x, rtol=1e-05, atol=1e-08) -> float:
    r"""Computes approximate GCD

    Parameters
    ----------
    x: ArrayLike
    rtol: float
    atol: float

    Returns
    -------
    float

    References
    ----------
    - <https://stackoverflow.com/q/45323619/9318372>
    """
    x = np.asanyarray(x)
    x = np.abs(x).flatten()

    @numba.njit
    def _float_gcd(x: np.ndarray) -> float:
        n = len(x)
        t = np.min(x)
        if n == 1:
            return float(x[0])
        if n == 2:
            while np.abs(x[1]) > rtol*t + atol:
                x[0], x[1] = x[1], x[0] % x[1]
            return float(x[0])
        # n >= 3:
        out = np.empty(2)
        out[0] = _float_gcd(x[:n//2])
        out[1] = _float_gcd(x[n//2:])
        return _float_gcd(out)

    return _float_gcd(x)


def is_regular(s: Series) -> bool:
    r"""Test if time series is regular

    Parameters
    ----------
    s: Series
        The timestamps

    Returns
    -------
    bool
    """
    Δt = np.diff(s)
    return bool(np.all(Δt == np.min(Δt)))


def is_quasiregular(s: Series) -> bool:
    r"""Test if time series is quasi-regular. By definition, this is the case if all timedeltas are
    integer multiples of the minimal, non-zero timedelta of the series.

    Parameters
    ----------
    s: Series

    Returns
    -------
    bool
    """
    Δt = np.diff(s)
    zero = np.array(0, dtype=Δt.dtype)
    Δt_min = np.min(Δt[Δt > zero])
    z = Δt / Δt_min
    return np.allclose(z, np.rint(z))


def time_gcd(s: Series):
    """Compute the greatest common divisor of datetime64/int/float data.

    Parameters
    ----------
    s: Series

    Returns
    -------
    gcd
    """
    Δt = np.diff(s)
    zero = np.array(0, dtype=Δt.dtype)
    Δt = Δt[Δt > zero]

    if np.issubdtype(Δt.dtype, np.timedelta64):
        Δt = Δt.astype('timedelta64[ns]').astype(int)
        gcd = np.gcd.reduce(Δt)
        return gcd.astype('timedelta64[ns]')
    if np.issubdtype(Δt.dtype, np.integer):
        return np.gcd.reduce(Δt)
    if np.issubdtype(Δt.dtype, np.floating):
        return float_gcd(Δt)

    raise NotImplementedError(F"Data type {Δt.dtype=} not understood")


def regularity_coefficient(s: Series) -> float:
    r"""Computes the regularity coefficient of a time series

    Parameters
    ----------
    s: Series

    Returns
    -------
    k:
        The regularity coefficient
    """
    gcd = time_gcd(s)
    Δt = np.diff(s)
    zero = np.array(0, dtype=Δt.dtype)
    Δt = Δt[Δt > zero]
    Δt_min = np.min(Δt)
    return Δt_min/gcd
