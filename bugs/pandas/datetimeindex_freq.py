import pandas as pd

data = ["2022-01-01T10:00:00", "2022-01-01T10:00:30", "2022-01-01T10:01:00"]
pd_series = pd.Series(data).astype("timestamp[s][pyarrow]")
pd_index = pd.Index(data).astype("timestamp[s][pyarrow]")
assert pd.infer_freq(pd_index.values) == "30s"  # ✅
assert pd.infer_freq(pd_series.values) == "30s"  # ✅
assert pd.infer_freq(pd_index) == "30s"  # ✅
assert pd.infer_freq(pd_series) == "30s"  # ❌
