#!/usr/bin/env python

from pandas import DataFrame, MultiIndex

index = MultiIndex.from_tuples([(0, 0), (0, 1)])
df = DataFrame(range(2), index=index)
key = (0, 0)
x = df.loc[key].copy()  # raises [union-attr] Item has no attribute "copy"
