#!/usr/bin/env python

import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
columns_types = {"a": "int", "b": "float"}
df = df.astype(columns_types)  # ✘ raises [arg-type]
# Argument 1 to "astype" of
# "DataFrame" has incompatible type "Series[Any]
