#!/usr/bin/env python

from typing import Generic, TypeVar

import numpy
import torch

T = TypeVar("T", numpy.ndarray, torch.Tensor)


class StandardScaler(Generic[T]):
    mean: T
    stdv: T

    def fit(self, data: T) -> None:
        if isinstance(data, torch.Tensor):
            self._fit_torch(data)
        elif isinstance(data, numpy.ndarray):
            self._fit_numpy(data)
        else:
            raise ValueError

    def _fit_torch(self: StandardScaler[torch.Tensor], data: torch.Tensor) -> None:
        self.mean = torch.mean(data)  # ✘ [assignment]
        self.stdv = torch.std(data)  # ✘ [assignment]

    def _fit_numpy(self, data: numpy.ndarray) -> None:
        self.mean = numpy.mean(data)
        self.stdv = numpy.std(data)
