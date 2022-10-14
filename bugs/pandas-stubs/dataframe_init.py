#!/usr/bin/env python

from collections.abc import Sequence

import pandas as pd

index = [3, 2, 7]
columns = ["a", "b", "c"]
df = pd.DataFrame(index=index, columns=columns)  # ✘ error: [call-overload]
print(df)
