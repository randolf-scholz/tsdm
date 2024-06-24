import pandas as pd

datetimes = [None, "2022-01-01T10:00:30", "2022-01-01T10:01:00"]
dt = pd.Index(datetimes, dtype="timestamp[ms][pyarrow]")

offset = pd.Timestamp("2022-01-01 10:00:30")
unit = pd.Index([pd.Timedelta(30, "s")], dtype="duration[ms][pyarrow]").item()

# %% encode to double[pyarrow]
encoded = (dt - offset) / unit  # min-max scaling
decoded = (encoded.round().astype(float) * unit + offset).astype(dt.dtype)

# compare original and decoded
pd.testing.assert_index_equal(dt, decoded, exact=True)  # ✅
# assert ((dt - dt) == 0).all()  # ✅
# assert ((decoded - decoded) == 0).all()  # ✅
# assert ((decoded - dt) == 0).all()  # ✅
assert ((dt - decoded) == 0).all()  # ❌ overflow ?!?!

# %% Try with different units
unit_a = pd.Timedelta(30, "s")  # <-- this one works!
unit_b = pd.Index([pd.Timedelta(30, "s")], dtype="duration[ms][pyarrow]").item()
assert unit_a == unit_b  # ✅
assert hash(unit_a) == hash(unit_b)  # ✅

# encode to double[pyarrow]
encoded_a = (dt - offset) / unit_a  # min-max scaling
encoded_b = (dt - offset) / unit_b  # min-max scaling
pd.testing.assert_index_equal(encoded_a, encoded_b, exact=True)  # ✅

# decode
decoded_a = (encoded_a.round().astype(float) * unit_a + offset).astype(dt.dtype)
decoded_b = (encoded_b.round().astype(float) * unit_b + offset).astype(dt.dtype)
pd.testing.assert_index_equal(decoded_a, decoded_b, exact=True)  # ✅
pd.testing.assert_index_equal(dt, decoded_a, exact=True)  # ✅
pd.testing.assert_index_equal(dt, decoded_b, exact=True)  # ✅
assert ((dt - decoded_a) == 0).all()  # ✅
assert ((dt - decoded_b) == 0).all()  # ❌ overflow ?!?!

# %% compute differences in pyarrow:
import pyarrow as pa

x = pa.Array.from_pandas(dt)
y = pa.Array.from_pandas(decoded)
pa.compute.subtract(x, y)  # ✅
pa.compute.subtract(y, x)  # ✅

# %%
pd.show_versions()
