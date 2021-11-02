"""Representation of Time Series Datasets."""
#
# from pandas import Series
# from torch import Tensor
#
#
# class TimeTensor(Tensor):
#
#     def __new__(cls, *args, **kwargs):
#         return super().__new__(cls, *args, **kwargs)
#
#     @property
#     def timeindex(self) -> Series:
#         ...
#
#     @property
#     def loc(self, value):
#         return self.timeindex.loc[value]
#
#
# class TimeSeriesDataset:
#     """
#     tuple[DataFrame]
#     tuple[DataFrame]
#     """
#
#
#
# class TimeSeriesCollection:
#     """
#     General TimeSeriesCollection:
#         Mapping[keys -> TimeSeriesDataset]
#     Regular TimeSeriesCollection:
#         All TimeSeriesDataset have the same schema and can be merged into a single table.
#     """
