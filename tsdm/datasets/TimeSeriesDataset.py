"""Representation of Time Series Datasets."""

from torch import Tensor

from pandas import Series

class TimeTensor(Tensor):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    @property
    def timeindex(self) -> Series:
        ...

    @property
    def loc(self, value):
        return self.timeindex.loc[value]


class TimeSeriesDataset:







