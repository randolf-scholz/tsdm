#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('config', "InteractiveShell.ast_node_interactivity='last_expr_or_assign'  # always print last expr.")
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import logging

logging.basicConfig(level=logging.INFO)


# In[2]:


import os

# enable JIT compilation - must be done before loading torch!
os.environ["PYTORCH_JIT"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
RUN_NAME = "Long+ResNet+AdamW+NRMSE"  # | input("enter name for run")


# In[3]:


from pathlib import Path
from time import perf_counter, time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torchinfo
from linodenet.models import LinODE, LinODECell, LinODEnet
from linodenet.projections.functional import skew_symmetric, symmetric
from pandas import DataFrame, Index, Series, Timedelta, Timestamp
from torch import Tensor, jit, nn, tensor
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

import tsdm
from tsdm.datasets import DATASETS
from tsdm.encoders.functional import time2float
from tsdm.logutils import (
    log_kernel_information,
    log_metrics,
    log_model_state,
    log_optimizer_state,
)
from tsdm.losses import LOSSES
from tsdm.tasks import KIWI_RUNS_TASK
from tsdm.util import grad_norm, multi_norm

# In[4]:


torch.backends.cudnn.benchmark = True


# # Initialize Task

# In[5]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NAN = tensor(float("nan"), dtype=DTYPE, device=DEVICE)
BATCH_SIZE = 128

# on average ca 30s between timestamps, i.e. 2 obs = 1min
# let's increase horizon. OBS: 240 = 2h, PRD = 120 = 1h
PRD_HORIZON = 120
OBS_HORIZON = 240
HORIZON = SEQLEN = OBS_HORIZON + PRD_HORIZON


# In[6]:


task = KIWI_RUNS_TASK(
    forecasting_horizon=PRD_HORIZON,
    observation_horizon=OBS_HORIZON,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=2048,
)

DATASET = task.dataset
ts = task.timeseries
md = task.metadata
NUM_PTS, NUM_DIM = ts.shape


# ## Initialize Loss & Metrics

# In[7]:


TASK_LOSS = task.test_metric.to(device=DEVICE)

metrics = {key: LOSSES[key] for key in ("ND", "NRMSE", "MSE", "MAE")}
# assert any(isinstance(TASK.test_metric, metric) for metric in metrics.values())
metrics = {key: LOSSES[key]() for key in ("ND", "NRMSE", "MSE", "MAE")} | {
    "WRMSE": TASK_LOSS
}

# LOSS = TASK_LOSS
# Let's try something else
LOSS = metrics["NRMSE"]


# In[8]:


task.loss_weights


# ## Initialize DataLoaders

# In[9]:


TRAINLOADERS = task.batchloaders
TRAINLOADER = TRAINLOADERS[(0, "train")]
EVALLOADERS = task.dataloaders


# ## Hyperparamters

# In[10]:


def join_dicts(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively join dict by composing keys with '/'."""
    result = {}
    for key, val in d.items():
        if isinstance(val, dict):
            result |= join_dicts(
                {f"{key}/{subkey}": item for subkey, item in val.items()}
            )
        else:
            result[key] = val
    return result


def add_prefix(d: dict[str, Any], /, prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{key}": item for key, item in d.items()}


# OPTIMIZER_CONIFG = {
#     "__name__": "SGD",
#     "lr": 0.001,
#     "momentum": 0,
#     "dampening": 0,
#     "weight_decay": 0,
#     "nesterov": False,
# }

# OPTIMIZER_CONFIG = {
#     "__name__": "Adam",
#     "lr": 0.01,
#     "betas": (0.9, 0.999),
#     "eps": 1e-08,
#     "weight_decay": 0,
#     "amsgrad": False,
# }


OPTIMIZER_CONFIG = {
    "__name__": "AdamW",
    "lr": 0.001,
    "betas": (0.9, 0.999),
    "eps": 1e-08,
    "weight_decay": 0.001,
    "amsgrad": False,
}


SYSTEM = {
    "__name__": "LinODECell",
    "input_size": int,
    "kernel_initialization": "skew-symmetric",
}

EMBEDDING = {
    "__name__": "ConcatEmbedding",
    "input_size": int,
    "hidden_size": int,
}
FILTER = {
    "__name__": "KalmanCell",
    "input_size": int,
    "hidden_size": int,
    "autoregressive": True,
    "activation": "Tanh",
    "use_rezero": True,
}

# FILTER = {
#     "__name__": "RecurrentCellFilter",
#     "concat": True,
#     "input_size": int,
#     "hidden_size": int,
#     "autoregressive": True,
#     "Cell": {
#         "__name__": "GRUCell",
#         "input_size": int,
#         "hidden_size": int,
#         "bias": True,
#         "device": None,
#         "dtype": None,
#     },
# }
from linodenet.models.encoders import ResNet, iResNet

# ENCODER = {"__name__": "ResNet", "__module__": "linodenet.models.encoders","input_size": int, "nblocks": 5, "rezero": True}
# DECODER = {"__name__": "ResNet", "__module__": "linodenet.models.encoders","input_size": int, "nblocks": 5, "rezero": True}

MODEL_CONFIG = {
    "__name__": "LinODEnet",
    "input_size": NUM_DIM,
    "hidden_size": 128,
    "embedding_type": "concat",
    "Filter": FILTER,
    "System": SYSTEM,
    "Encoder": ResNet.DEFAULT_HP,
    "Decoder": ResNet.DEFAULT_HP,
    "Embedding": EMBEDDING,
}


HPARAMS = join_dicts(
    {
        "Optimizer": OPTIMIZER_CONFIG,
        "Model": MODEL_CONFIG,
    }
)


# In[11]:


model = ResNet(input_size=12, rezero=True)
torchinfo.summary(model, depth=4)


# ## Initialize Model

# In[12]:


MODEL = LinODEnet
model = MODEL(**MODEL_CONFIG)
model.to(device=DEVICE, dtype=DTYPE)
torchinfo.summary(model)


# ### Initialized Kernel statistics

# In[13]:


expA = torch.matrix_exp(model.kernel)
for o in (-np.infty, -2, -1, 1, 2, np.infty, "fro", "nuc"):
    val = torch.linalg.matrix_norm(model.kernel, ord=o).item()
    val2 = torch.linalg.matrix_norm(expA, ord=o).item()
    o = str(o)
    print(f"{o=:6s}\t {val=:10.6f} \t {val2=:10.6f}")


# ## Initalize Optimizer

# In[14]:


from tsdm.optimizers import OPTIMIZERS
from tsdm.util import initialize_from

# In[15]:


OPTIMIZER_CONFIG |= {"params": model.parameters()}
optimizer = initialize_from(OPTIMIZERS, **OPTIMIZER_CONFIG)


# ## Utility functions

# In[16]:


batch = next(iter(TRAINLOADER))
T, X = batch
targets = X[..., OBS_HORIZON:, task.targets.index].clone()
# assert targets.shape == (BATCH_SIZE, PRD_HORIZON, len(TASK.targets))

inputs = X.clone()
inputs[:, OBS_HORIZON:, task.targets.index] = NAN
inputs[:, OBS_HORIZON:, task.observables.index] = NAN
# assert inputs.shape == (BATCH_SIZE, HORIZON, NUM_DIM)


# In[17]:


targets = X[..., OBS_HORIZON:, task.targets.index].clone()
targets.shape


# In[18]:


def prep_batch(batch: tuple[Tensor, Tensor]):
    """Get batch and create model inputs and targets"""
    T, X = batch
    T = T.cuda(non_blocking=True)
    X = X.cuda(non_blocking=True)

    targets = X[..., OBS_HORIZON:, task.targets.index].clone()
    # assert targets.shape == (BATCH_SIZE, PRD_HORIZON, len(TASK.targets))

    inputs = X.clone()
    inputs[:, OBS_HORIZON:, task.targets.index] = NAN
    inputs[:, OBS_HORIZON:, task.observables.index] = NAN
    # assert inputs.shape == (BATCH_SIZE, HORIZON, NUM_DIM)
    return T, inputs, targets


def get_all_preds(model, dataloader):
    y, yhat = [], []
    for batch in (pbar := tqdm(dataloader, leave=False)):
        with torch.no_grad():
            model.zero_grad()
            times, inputs, targets = prep_batch(batch)
            outputs = model(times, inputs)
            predics = outputs[:, OBS_HORIZON:, task.targets.index]
            loss = LOSS(targets, predics)
            y.append(targets)
            yhat.append(predics)
        if pbar.n == 5:
            break

    targets, predics = torch.cat(y, dim=0), torch.cat(yhat, dim=0)
    mask = torch.isnan(targets)
    targets[mask] = torch.tensor(0.0)
    predics[mask] = torch.tensor(0.0)
    # scale = 1/torch.mean(mask.to(dtype=torch.float32))
    # targets *= scale
    # predics *= scale
    return targets, predics


# ## Logging Utilities

# In[19]:


from tsdm.logutils import compute_metrics


def log_all(i, model, writer, optimizer):
    kernel = model.system.kernel.clone().detach().cpu()
    log_kernel_information(i, writer, kernel, histograms=True)
    log_optimizer_state(i, writer, optimizer, histograms=True)


def log_hparams(i, writer, *, metric_dict, hparam_dict):
    hparam_dict |= {"epoch": i}
    metric_dict = add_prefix(metric_dict, "hparam")
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)


print("WARMUP")

t = torch.randn(3, NUM_DIM).to(DEVICE)
x = torch.randn(3, 1, NUM_DIM).to(device=DEVICE)
y = model(t, x)
torch.linalg.norm(y).backward()
model.zero_grad()


# In[20]:


RUN_START = tsdm.util.now()
CHECKPOINTDIR = Path(
    f"checkpoints/{MODEL.__name__}/{DATASET.name}/{RUN_NAME}/{RUN_START}"
)
CHECKPOINTDIR.mkdir(parents=True, exist_ok=True)
LOGGING_DIR = f"runs/{MODEL.__name__}/{DATASET.name}/{RUN_NAME}/{RUN_START}"
writer = SummaryWriter(LOGGING_DIR)


# # Training

# In[ ]:


i = -1
epoch = 1

with torch.no_grad():
    # log optimizer state first !!!
    # log_optimizer_state(epoch, writer, optimizer, histograms=True)
    log_kernel_information(epoch, writer, model.system.kernel, histograms=True)

    for key in ((0, "train"), (0, "test")):
        dataloader = EVALLOADERS[key]
        y, ŷ = get_all_preds(model, dataloader)
        assert torch.isfinite(y).all()
        log_metrics(
            epoch, writer, metrics=metrics, targets=y, predics=ŷ, prefix=key[1]
        )


for _ in (epochs := trange(100)):
    epoch += 1
    batching_time = perf_counter()
    for batch in (batches := tqdm(TRAINLOADER)):
        batching_time = perf_counter() - batching_time
        i += 1
        # Optimization step
        model.zero_grad(set_to_none=True)
        times, inputs, targets = prep_batch(batch)

        forward_time = perf_counter()
        outputs = model(times, inputs)
        forward_time = perf_counter() - forward_time

        predics = outputs[:, OBS_HORIZON:, task.targets.index]

        # get rid of nan-values in teh targets.
        mask = torch.isnan(targets)
        targets[mask] = torch.tensor(0.0)
        predics[mask] = torch.tensor(0.0)

        # # compensate NaN-Value with upscaling
        # scale = 1/torch.mean(mask.to(dtype=torch.float32))
        # targets *= scale
        # predics *= scale

        loss = LOSS(targets, predics)

        backward_time = perf_counter()
        loss.backward()
        backward_time = perf_counter() - backward_time

        optimizer.step()

        # batch logging
        logging_time = time()
        with torch.no_grad():
            if torch.any(torch.isnan(loss)):
                raise RuntimeError("NaN-value encountered!!")

            log_metrics(
                i,
                writer,
                metrics=metrics,
                targets=targets.clone(),
                predics=predics.clone(),
                prefix="batch",
            )
            log_optimizer_state(i, writer, optimizer, prefix="batch")

            # lval = loss.clone().detach().cpu().numpy()
            # gval = grad_norm(list(model.parameters())).clone().detach().cpu().numpy()

        logging_time = time() - logging_time

        batches.set_postfix(
            # loss=f"{lval:.2e}",
            # gnorm=f"{gval:.2e}",
            Δt_forward=f"{forward_time:.1f}",
            Δt_backward=f"{backward_time:.1f}",
            Δt_logging=f"{logging_time:.1f}",
            Δt_batching=f"{batching_time:.1f}",
        )
        batching_time = perf_counter()

    with torch.no_grad():
        # log optimizer state first !!!
        log_optimizer_state(epoch, writer, optimizer, histograms=True)
        log_kernel_information(epoch, writer, model.system.kernel, histograms=True)

        for key in ((0, "train"), (0, "test")):
            dataloader = EVALLOADERS[key]
            y, ŷ = get_all_preds(model, dataloader)
            metric_values = compute_metrics(metrics, targets=y, predics=ŷ)
            log_metrics(
                epoch, writer, metrics=metrics, values=metric_values, prefix=key[1]
            )
            # log_hparams(epoch, writer, metric_dict=metric_values, hparam_dict=HPARAMS)

        # Model Checkpoint
        torch.jit.save(model, CHECKPOINTDIR.joinpath(f"{MODEL.__name__}-{epochs.n}"))
        torch.save(
            {
                "optimizer": optimizer,
                "epoch": epoch,
                "batch": i,
            },
            CHECKPOINTDIR.joinpath(f"{optimizer.__class__.__name__}-{epochs.n}"),
        )


# In[ ]:


raise StopIteration


# # Post Training Analysis

# In[ ]:


buffers = dict(model.named_buffers())
set(buffers.keys())


# In[ ]:


timedeltas = model.timedeltas.detach().cpu()
xhat_pre = model.xhat_pre.detach().cpu()
xhat_post = model.xhat_post.detach().cpu()
zhat_pre = model.zhat_pre.detach().cpu()
zhat_post = model.zhat_post.detach().cpu()
xhat_pre.shape, xhat_post.shape, zhat_pre.shape, zhat_post.shape


# ## Relative size change xhat_pre ⟶ xhat_post

# In[ ]:


get_ipython().run_line_magic("matplotlib", "inline")
plt.style.use("bmh")

BATCH_DIM, LEN, DIM = tuple(xhat_pre.shape)
n, m = model.input_size, model.hidden_size


def gmean(x, dim=(), p=2):
    """Geometric mean"""
    return torch.exp(torch.mean(torch.log(torch.abs(x) ** p), dim=dim) ** (1 / p))


predata = xhat_pre
postdata = xhat_post

xpretotalmag = torch.mean(
    torch.linalg.norm(xhat_pre, dim=-1) / torch.linalg.norm(xhat_pre[:, [0]], dim=-1),
    dim=0,
).squeeze()

xpsttotalmag = torch.mean(
    torch.linalg.norm(xhat_post, dim=-1) / torch.linalg.norm(xhat_post[:, [0]], dim=-1),
    dim=0,
).squeeze()

zpretotalmag = torch.mean(
    torch.linalg.norm(zhat_pre, dim=-1) / torch.linalg.norm(zhat_pre[:, [0]], dim=-1),
    dim=0,
).squeeze()

zpsttotalmag = torch.mean(
    torch.linalg.norm(zhat_post, dim=-1) / torch.linalg.norm(zhat_post[:, [0]], dim=-1),
    dim=0,
).squeeze()

xpremag = torch.mean(
    torch.linalg.norm(xhat_pre[..., 1:, :], dim=-1)
    / torch.linalg.norm(xhat_pre[..., :-1, :], dim=-1),
    dim=0,
)
xpstmag = torch.mean(
    torch.linalg.norm(xhat_post[..., 1:, :], dim=-1)
    / torch.linalg.norm(xhat_post[..., :-1, :], dim=-1),
    dim=0,
)
zpremag = torch.mean(
    torch.linalg.norm(zhat_pre[..., 1:, :], dim=-1)
    / torch.linalg.norm(zhat_pre[..., :-1, :], dim=-1),
    dim=0,
)
zpstmag = torch.mean(
    torch.linalg.norm(zhat_post[..., 1:, :], dim=-1)
    / torch.linalg.norm(zhat_post[..., :-1, :], dim=-1),
    dim=0,
)

system_mag = torch.linalg.norm(zhat_pre[:, 1:], dim=-1) / torch.linalg.norm(
    zhat_post[:, :-1], dim=-1
)
system_mag = torch.cat([torch.ones(BATCH_DIM, 1), system_mag], dim=-1)
combine_mag = torch.linalg.norm(zhat_post, dim=-1) / torch.linalg.norm(zhat_pre, dim=-1)
# system_mag = torch.cat([torch.ones(BATCH_DIM, 1), system_mag], dim=-1)
decoder_mag = gmean(xhat_pre, dim=-1) / gmean(zhat_pre, dim=-1)
filter_mag = gmean(xhat_post, dim=-1) / gmean(xhat_pre, dim=-1)
encoder_mag = gmean(zhat_post, dim=-1) / gmean(xhat_post, dim=-1)

filter_mag = torch.mean(filter_mag, dim=0)
system_mag = torch.mean(system_mag, dim=0)
combine_mag = torch.mean(combine_mag, dim=0)
decoder_mag = torch.mean(decoder_mag, dim=0)
encoder_mag = torch.mean(encoder_mag, dim=0)

fig, ax = plt.subplots(
    ncols=4, nrows=3, figsize=(12, 8), sharey="row", constrained_layout=True
)

ax[0, 0].semilogy(xpretotalmag)
ax[0, 0].set_title(r"Rel. Magnitude change $\hat{x}_0  \rightarrow \hat{x}_{t+1}  $")
ax[0, 1].semilogy(xpsttotalmag)
ax[0, 1].set_title(r"Rel. Magnitude change $\hat{x}_0' \rightarrow \hat{x}_{t+1}' $")
ax[0, 2].semilogy(zpretotalmag)
ax[0, 2].set_title(r"Rel. Magnitude change $\hat{z}_0  \rightarrow \hat{z}_{t+1}  $")
ax[0, 3].semilogy(zpsttotalmag)
ax[0, 3].set_title(r"Rel. Magnitude change $\hat{z}_0' \rightarrow \hat{z}_{t+1}' $")

ax[1, 0].semilogy(xpremag)
ax[1, 0].set_title(r"Rel. Magnitude change $\hat{x}_t  \rightarrow \hat{x}_{t+1}  $")
ax[1, 1].semilogy(xpstmag)
ax[1, 1].set_title(r"Rel. Magnitude change $\hat{x}_t' \rightarrow \hat{x}_{t+1}' $")
ax[1, 2].semilogy(zpremag)
ax[1, 2].set_title(r"Rel. Magnitude change $\hat{z}_t  \rightarrow \hat{z}_{t+1}  $")
ax[1, 3].semilogy(zpstmag)
ax[1, 3].set_title(r"Rel. Magnitude change $\hat{z}_t' \rightarrow \hat{z}_{t+1}' $")

ax[2, 0].semilogy(decoder_mag)
ax[2, 0].set_title(r"Rel. magnitude change $\hat{z}_t  \rightarrow \hat{x}_t$")
ax[2, 1].semilogy(filter_mag)
ax[2, 1].set_title(r"Relative magnitude change $\hat{x}_t  \rightarrow \hat{x}_t'$")
# ax[1, 2].semilogy(encoder_mag)
# ax[1, 2].set_title(r"Relative magnitude change $\hat{x}_t' \rightarrow \hat{z}_t'$")
ax[2, 2].semilogy(encoder_mag)
ax[2, 2].set_title(r"Rel. magnitude change $\hat{x}_t' \rightarrow \hat{z}_t'$")
ax[2, 3].semilogy(system_mag)
ax[2, 3].set_title(r"Rel. magnitude change $\hat{x}_t' \rightarrow \hat{z}_t'$")
# ax[2, 3].semilogy(combine_mag)
# ax[2, 3].set_title(r"Rel. magnitude change $\hat{z}_t \rightarrow \hat{z}_{t}'$")
# ax[2, 0].set_yscale("log")
fig.savefig(f"{RUN_NAME}_encoder_stats_post_training.pdf")


# In[ ]:


# # distribution plots

# In[ ]:


xhat_pre_mean = torch.mean(xhat_pre, dim=-1).mean(dim=0)
xhat_pre_stdv = torch.std(xhat_pre, dim=-1).mean(dim=0)
xhat_post_mean = torch.mean(xhat_post, dim=-1).mean(dim=0)
xhat_post_stdv = torch.std(xhat_post, dim=-1).mean(dim=0)
zhat_pre_mean = torch.mean(zhat_pre, dim=-1).mean(dim=0)
zhat_pre_stdv = torch.std(zhat_pre, dim=-1).mean(dim=0)
zhat_post_mean = torch.mean(zhat_post, dim=-1).mean(dim=0)
zhat_post_stdv = torch.std(zhat_post, dim=-1).mean(dim=0)

tuples = [
    (r"$\hat{x}$", xhat_pre_mean, xhat_pre_stdv),
    (r"$\hat{x}'$", xhat_post_mean, xhat_post_stdv),
    (r"$\hat{z}$", zhat_pre_mean, zhat_pre_stdv),
    (r"$\hat{x}'$", zhat_post_mean, zhat_post_stdv),
]

S = np.arange(len(xhat_pre_mean))


# In[ ]:


fig, axes = plt.subplots(
    nrows=2, ncols=2, constrained_layout=True, figsize=(8, 5), sharex=True, sharey=True
)

for ax, (key, mean, std) in zip(axes.flatten(), tuples):
    color = next(ax._get_lines.prop_cycler)["color"]
    ax.plot(S, mean, color=color)
    ax.fill_between(S, mean + std, mean - std, alpha=0.3)
    ax.set_title(key)
    ax.set_yscale("symlog")


# In[ ]:


xhat_pre[0, 0]


# In[ ]:


xhat_pre[0, 1]


# In[ ]:


xhat_pre[0, 2]


# In[ ]:


xhat_pre[0, -1]


# In[ ]:


dummy = torch.randn(10_000, m, device="cuda")
dummy2 = model.encoder(dummy)
dummy1 = torch.linalg.norm(dummy, dim=-1) / m
dummy2 = torch.linalg.norm(dummy2, dim=-1) / m
chg = (dummy2 / dummy1).clone().detach().cpu().numpy()
plt.hist(chg, bins="auto")


# In[ ]:


expA = torch.matrix_exp(model.kernel)

for o in (-np.infty, -2, -1, 1, 2, np.infty, "fro", "nuc"):
    val = torch.linalg.matrix_norm(model.kernel, ord=o).item()
    val2 = torch.linalg.matrix_norm(expA, ord=o).item()
    o = str(o)
    print(f"{o=:6s}\t {val=:10.6f} \t {val2=:10.6f}")


# In[ ]:


from matplotlib import cm

mat = model.kernel.clone().detach().cpu()
mat = 0.5 + (mat - mat.mean()) / (6 * mat.std())
# mat = kernel.clip(0, 1)
colormap = cm.get_cmap("seismic")
mat = colormap(mat)
plt.imshow(mat)


# # Profiling

# In[ ]:


from torch.profiler import ProfilerActivity, profile, record_function

# In[ ]:


with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    model(times, inputs)


# In[ ]:


print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


# In[ ]:


print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


# In[ ]:
