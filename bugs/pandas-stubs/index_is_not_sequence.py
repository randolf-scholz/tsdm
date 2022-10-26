#!/usr/bin/env python

from collections.abc import Sequence

import pandas as pd

df = pd.DataFrame([1, 2, 3])
index: Sequence = df.index

# assert isinstance(index, Sequence)
print(index)
# print(index.__dir__())
print(set([1, 2, 3].__dir__()) - set(index.__dir__()))
