#!/usr/bin/env python

# # Title

# In[1]:


# %config InteractiveShell.ast_node_interactivity='last_expr_or_assign'  # always print last expr.
# %config InlineBackend.figure_format = 'svg'
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# import logging
# logging.basicConfig(level=logging.INFO)


# In[2]:


import gc
from contextlib import ContextDecorator
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from pandas import DataFrame, MultiIndex, Series
from tqdm.auto import tqdm


class Timer(ContextDecorator):
    __slots__ = ("start", "stop", "duration")  # faster access to the attributes
    start: float
    stop: float
    duration: float

    def __enter__(self):
        gc.collect()  # make space
        gc.disable()  # disable gc
        torch.cuda.synchronize()  # wait for cuda to finish
        self.start = perf_counter()

    def __exit__(self, *exc):
        torch.cuda.synchronize()  # wait for cuda to finish
        self.stop = perf_counter()
        self.duration = self.stop - self.start
        gc.enable()  # enable gc
        gc.collect()  # make space


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)
rng = np.random.default_rng()


# In[3]:


reductions = []
for a, b in product("ijkl", "ijkl"):
    if a == b:
        continue
    reduction = f"{a}{b},ijkl->" + "ijkl".replace(a, "").replace(b, "")
    reductions.append(reduction)

FRAMEWORKS = ["torch"]
DTYPES = ["float32"]
SIZES = [128]
DEVICES = [torch.device("cpu"), torch.device("cuda")]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TORCH_DTYPES = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
}

columns = Series(reductions, name="reduction")
index = MultiIndex.from_product(
    [SIZES, DTYPES, FRAMEWORKS], names=["size", "dtype", "framework"]
)
results = DataFrame(np.nan, index=index, columns=columns, dtype=float)
results.to_csv("einsum_slow.csv")


# In[ ]:

timer = Timer()


# torch_results
for size in tqdm(SIZES):
    _mat1 = torch.randn((size, size, size, size), device=DEVICE)
    _mat2 = torch.randn((size, size), device=DEVICE)

    for dtype in tqdm(DTYPES, leave=False):
        mat1 = _mat1.to(dtype=TORCH_DTYPES[dtype])
        mat2 = _mat2.to(dtype=TORCH_DTYPES[dtype])

        for reduction in tqdm(reductions, leave=False):
            with timer:
                for k in range(100):
                    torch.einsum(reduction, mat2, mat1)
                torch.cuda.synchronize()
            results.loc[(size, dtype, "torch"), reduction] = timer.duration


# In[ ]:


# # numpy results
# for size in tqdm(sizes):
#     _mat1 = np.random.normal(size=(size, size, size, size))
#     _mat2 = np.random.normal(size=(size, size))
#
#     for dtype in tqdm(dtypes, leave=False):
#         mat1 = _mat1.astype(dtype)
#         mat2 = _mat2.astype(dtype)
#
#         for reduction in tqdm(reductions, leave=False):
#             with timer:
#                 np.einsum(reduction, mat2, mat1, optimize=False)
#             results.loc[(size, dtype, "numpy"), reduction] = timer.duration
#

# In[ ]:


df = results.round(3).sort_values(["size", "dtype", "framework"])
df = df.T
print(df.columns)
df = df.sort_values(by=df.columns.tolist())
path = Path(__file__).parent
with open(path / "einsum_slow.fwf", "w") as file:
    print(f"Writing results to {path / 'einsum_slow.fwf'}")
    file.write(df.to_string())
df.to_csv(path / "einsum_slow.csv")


# In[ ]:
