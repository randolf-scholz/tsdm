#!/usr/bin/env python

import pandas as pd

df = (
    pd.DataFrame([("A", 1), ("B", 2), ("C", 3)], columns=["var", "val"])
    .astype({"var": "string", "val": "float32"})
    .astype({"var": "category", "val": "float32"})
)

# write and reload as parquet with pyarrow backend
df.to_parquet("demo.parquet")
df = pd.read_parquet("demo.parquet", dtype_backend="pyarrow")
print(df.dtypes)  # var is now dictionary[int32,string]
print(df)
x = df.pivot(columns=["var"], values=["val"])  # ✘ ArrowNotImplementedError
print(x)
print(x.dtypes)
