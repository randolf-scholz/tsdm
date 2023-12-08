"""Implementation of Picewise Linear Encoding (PLE).

References:
    On Embeddings for Numerical Features in Tabular Deep Learning
"""

# from tsdm.encoders.base import BaseEncoder
# from tsdm.types.variables import any2_var as S, any_var as T
#
#
# class PLE(BaseEncoder):
#     """Piecewise Linear Encoding (PLE)."""
#
#     bins: list[float]
#
#     @property
#     def requires_fit(self) -> bool:
#         return True
#
#     def __init__(self, bins: list[float] = None) -> None:
#         """Bins."""
#         self.bins = bins
#
#     def encode(self, data: T, /) -> S:
#         pass
#
#     def decode(self, data: S, /) -> T:
#         pass
