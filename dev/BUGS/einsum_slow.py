#!/usr/bin/env python
# %%
# %config InteractiveShell.ast_node_interactivity='last_expr_or_assign'

import gc
from contextlib import ContextDecorator
from itertools import combinations, permutations, product
from pathlib import Path
from time import perf_counter
from typing import Iterator

import jax.numpy as jnp
import numpy as np
import torch
from pandas import DataFrame, MultiIndex, Series
from tqdm.auto import tqdm

np.set_printoptions(precision=4, floatmode="fixed", suppress=True)
rng = np.random.default_rng()


# %%
class PerformanceTimer(ContextDecorator):
    __slots__ = ("start", "stop", "duration")  # faster access to the attributes
    start: float
    stop: float
    duration: float

    def __enter__(self):
        torch.cuda.synchronize()  # wait for cuda to finish
        gc.collect()  # run gc to free up memory
        gc.disable()  # disable gc
        self.start = perf_counter()

    def __exit__(self, *exc):
        torch.cuda.synchronize()  # wait for cuda to finish
        self.stop = perf_counter()
        self.duration = self.stop - self.start
        gc.enable()  # enable gc
        gc.collect()  # run gc to free up memory


# %%
def generate_reductions(n: int, m: int) -> Iterator[str]:
    """Generate all possible einsum-reductions over m-axes of an n-dim tensor."""
    assert m <= n, "m must be smaller or equal to n"
    operator = "ijklmnopqrstuv"[:n]
    for subset in combinations(operator, m):
        for operand in permutations(subset):
            # remove subset from letters
            result = "".join(c for c in operator if c not in operand)
            yield f"{operator}, {''.join(operand)} -> {result}"


REDUCTIONS = list(generate_reductions(4, 2))


# %%
FRAMEWORKS = ["torch", "jax"]
SIZES = [128]

FWORK_DTYPES = {
    "torch": {
        "int32": torch.int32,
        "int64": torch.int64,
        "float32": torch.float32,
        "float64": torch.float64,
    },
    "jax": {},
    "numpy": NotImplemented,
    "tensorflow": NotImplemented,
    "cupy": NotImplemented,
    "mxnet": NotImplemented,
}
CHOSEN_DTYPES = ["float32"]
DTYPES = {
    fwork: [FWORK_DTYPES[fwork][dtype] for dtype in CHOSEN_DTYPES]
    for fwork in FRAMEWORKS
}


FWORK_DEVICES = {
    "torch": [torch.device(x) for x in _DEVICES],
    "jax": NotImplemented,
    "numpy": NotImplemented,
    "tensorflow": NotImplemented,
    "cupy": NotImplemented,
    "mxnet": NotImplemented,
}
CHOSEN_DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
DEVICES = {
    fwork: [FWORK_DEVICES[fwork][device] for device in CHOSEN_DEVICES]
    for fwork in FRAMEWORKS
}


columns = Series(REDUCTIONS, name="reduction")
index = MultiIndex.from_product(
    [SIZES, DTYPES, FRAMEWORKS], names=["size", "dtype", "framework"]
)
results = DataFrame(np.nan, index=index, columns=columns, dtype=float)
results.to_csv("einsum_slow.csv")


# %%
timer = PerformanceTimer()


# %%
def torch_experiment(timer, mat1, mat2, results, repetitions=100) -> None:
    dtypes = DTYPES["torch"]
    devices = DEVICES["torch"]

    for dtype, device in tqdm(product(dtypes, devices), leave=False):
        operator = torch.tensor(mat1, dtype=dtype, device=device)
        operand = torch.tensor(mat2, dtype=dtype, device=device)

        for reduction in tqdm(REDUCTIONS, leave=False):
            with timer:
                for k in range(repetitions):
                    torch.einsum(reduction, operator, operand)
            results.loc[(size, dtype, "torch"), reduction] = timer.duration


# %% [markdown]
# ## torch results


# %%
for size in tqdm(SIZES):
    mat1 = torch.randn((size, size, size, size), device=torch.device(DEVICE))
    mat2 = torch.randn((size, size), device=torch.device(DEVICE))

    for dtype in tqdm(DTYPES, leave=False):
        operator = mat1.to(dtype=TORCH_DTYPES[dtype])
        operand = mat2.to(dtype=TORCH_DTYPES[dtype])

        for reduction in tqdm(REDUCTIONS, leave=False):
            with timer:
                for k in range(100):
                    torch.einsum(reduction, operator, operand)
            results.loc[(size, dtype, "torch"), reduction] = timer.duration


# %% [markdown]
# ## jax results

# %%
for size in tqdm(SIZES):
    mat1 = np.random.normal(size=(size, size, size, size))
    mat2 = np.random.normal(size=(size, size))

    for dtype in tqdm(DTYPES, leave=False):
        operator = jnp.array(mat1.astype(dtype))
        operand = jnp.array(mat2.astype(dtype))

        for reduction in tqdm(REDUCTIONS, leave=False):
            with timer:
                jnp.einsum(reduction, operator, operand, optimize=True)
            results.loc[(size, dtype, "jax"), reduction] = timer.duration

# %% [markdown]
# ## numpy results

# %%
for size in tqdm(SIZES):
    mat1 = np.random.normal(size=(size, size, size, size))
    mat2 = np.random.normal(size=(size, size))

    for dtype in tqdm(DTYPES, leave=False):
        operator = mat1.astype(dtype)
        operand = mat2.astype(dtype)

        for reduction in tqdm(REDUCTIONS, leave=False):
            with timer:
                np.einsum(reduction, mat2, mat1, optimize=False)
            results.loc[(size, dtype, "numpy"), reduction] = timer.duration

# %% [markdown]
# # Results

# %%
df = results.round(3).sort_values(["size", "dtype", "framework"])
df = df.T
df = df.sort_values(by=df.columns.tolist())


# %%
path = Path.cwd()
with open(path / "einsum_slow.fwf", "w") as file:
    print(f"Writing results to {path / 'einsum_slow.fwf'}")
    file.write(df.to_string())
df.to_csv(path / "einsum_slow.csv")
