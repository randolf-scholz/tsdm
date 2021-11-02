r"""Some default mechanisms for storing collections of DataFrames in HDF5.

Mental Model
------------

Most data we deal with will be processed using Linear Algebra in one way or another.
The fundamental concept of applied linear algebra is that of an inner product, or
more specifically that of a Hilbert space. There are two fundamental operations to
create a larger Hilbert space from two given Hilbert spaces

- Sum (aka direct sum):   U⊕V
- Product (aka tensor product):

We assume that the dataset can be described in the form

iterable[tuple[tensor]]
"""

from __future__ import annotations

import logging

# __all__ = []


__logger__ = logging.getLogger(__name__)
