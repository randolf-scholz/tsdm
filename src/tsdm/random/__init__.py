r"""Statistical functions for random variables."""

__all__ = [
    # Sub-Packages
    "generators",
    "samplers",
    "distributions",
]


lazy from . import distributions, generators, samplers
