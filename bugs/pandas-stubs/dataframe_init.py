#!/usr/bin/env python


import pandas as pd

index: tuple[int, int, int] = [3, 2, 7]
columns = ["a", "b", "c"]

# index = "abc"

df = pd.DataFrame(index=index, columns=columns)  # ✘ error: [call-overload]
print(df)
