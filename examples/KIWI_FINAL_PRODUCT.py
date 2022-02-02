#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('config', "InteractiveShell.ast_node_interactivity='last_expr_or_assign'  # always print last expr.")
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import logging

import os
from pathlib import Path
from time import perf_counter, time
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torchinfo
from linodenet.models import LinODE, LinODECell, LinODEnet
from linodenet.models.filters import SequentialFilter
from linodenet.projections.functional import skew_symmetric, symmetric
from pandas import DataFrame, Index, Series, Timedelta, Timestamp
from torch import Tensor, jit, nn, tensor
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import BatchSampler, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import tsdm
from tsdm.datasets import DATASETS
from tsdm.encoders.functional import time2float
from tsdm.encoders.modular import *
from tsdm.logutils import (
    log_kernel_information,
    log_metrics,
    log_model_state,
    log_optimizer_state,
)
from tsdm.losses import LOSSES
from tsdm.tasks import KIWI_FINAL_PRODUCT
from tsdm.util import grad_norm, multi_norm
from tsdm.util.strings import *
from tsdm.optimizers import LR_SCHEDULERS, OPTIMIZERS
from tsdm.util import initialize_from

from torchinfo import summary

from tsdm.models import SetFuncTS
from tsdm.logutils import compute_metrics

def run():

    # logging.basicConfig(level=logging.INFO)


    # In[2]:



    # enable JIT compilation - must be done before loading torch!
    os.environ["PYTORCH_JIT"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    RUN_NAME = "Final_Product"  # | input("enter name for run")


    # In[3]:





    # In[4]:


    torch.backends.cudnn.benchmark = True


    # # Initialize Task

    # In[5]:


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32
    NAN = tensor(float("nan"), dtype=DTYPE, device=DEVICE)
    BATCH_SIZE = 256
    TARGET = "Fluo_GFP"
    SPLIT = (0, "train")

    task = KIWI_FINAL_PRODUCT(
        train_batch_size=BATCH_SIZE,
        eval_batch_size=2048,
        target=TARGET,
    )

    DATASET = task.dataset
    ts = task.timeseries
    md = task.metadata
    NUM_PTS, NUM_DIM = ts.shape


    # ## Initialize Encoder

    # In[6]:


    ts, md = task.splits[SPLIT]


    encoder = ChainedEncoder(
        TensorEncoder(device="cuda", names=("time", "value", "index")),
        DataFrameEncoder(
            column_encoders={
                "value": IdentityEncoder(),
                tuple(ts.columns): FloatEncoder("float32"),
            },
            index_encoders=MinMaxScaler() @ DateTimeEncoder(unit="h"),
        ),
        TripletEncoder(sparse=True),
        Standardizer(),
    )
    encoder.fit(ts.reset_index([0, 1], drop=True))
    task.target_idx = task.timeseries.columns.get_loc(task.target)
    target_encoder = (
        TensorEncoder(device="cuda") @ FloatEncoder() @ encoder[-1][task.target_idx]
    )


    # ## Define Batching Function

    # In[7]:


    class Batch(NamedTuple):
        index: Tensor
        timeseries: Tensor
        metadata: Tensor
        targets: Tensor
        encoded_targets: Tensor

        def __repr__(self):
            return repr_mapping(
                self._asdict(), title=self.__class__.__name__, repr_fun=repr_array
            )


    @torch.no_grad()
    def mycollate(batch: list):
        index = []
        timeseries = []
        metadata = []
        targets = []
        encoded_targets = []

        for idx, (ts_data, (md_data, target)) in batch:
            index.append(torch.tensor(idx[0]))
            timeseries.append(encoder.encode(ts_data))
            metadata.append(md_data)
            targets.append(target)
            encoded_targets.append(target_encoder.encode(target))

        index = torch.stack(index)
        targets = pandas.concat(targets)
        encoded_targets = torch.concat(encoded_targets)

        return Batch(index, timeseries, metadata, targets, encoded_targets)


    # ## Initialize Loss & Metrics

    # In[8]:


    LOSS = task.test_metric.to(device=DEVICE)

    # metrics = {key: LOSSES[key] for key in ("ND", "NRMSE", "MSE", "MAE")}
    # # assert any(isinstance(TASK.test_metric, metric) for metric in metrics.values())
    # metrics = {key: LOSSES[key]() for key in ("ND", "NRMSE", "MSE", "MAE")} | {
    #     "WRMSE": TASK_LOSS
    # }

    # # LOSS = TASK_LOSS
    # # Let's try something else
    # LOSS = metrics["NRMSE"]


    # ## Initialize DataLoaders

    # In[9]:


    TRAINLOADERS = task.batchloaders
    TRAINLOADER = TRAINLOADERS[SPLIT]
    EVALLOADERS = task.dataloaders
    EVALLOADER = EVALLOADERS[SPLIT]

    # Add the encoding
    # TRAINLOADER.collate_fn = mycollate


    # ## Hyperparamters

    # In[10]:


    OPTIMIZER_CONFIG = {
        "__name__": "AdamW",
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0.001,
        "amsgrad": False,
    }

    LR_SCHEDULER_CONFIG = {
        "__name__": "ReduceLROnPlateau",
        "mode": "min",
        # (str) – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        "factor": 0.1,
        # (float) – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        "patience": 10,
        # (int) – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        "threshold": 0.0001,
        # (float) – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        "threshold_mode": "rel",
        # (str) – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        "cooldown": 0,
        # (int) – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        "min_lr": 1e-08,
        # (float or list) – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        "eps": 1e-08,
        # (float) – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
        "verbose": True
        # (bool) – If True, prints a message to stdout for each update. Default: False.
    }


    # ## Model

    # In[11]:


    MODEL = SetFuncTS
    model = MODEL(17, 1)
    model.to(device=DEVICE, dtype=DTYPE)
    summary(model)


    # ### Warmup - test forward / backward pass

    # In[27]:


    del TRAINLOADER
    TRAINLOADER = task.get_dataloader(
        (0, "train"), batch_size=64, collate_fn=mycollate, num_workers=2, pin_memory=False
    )


    # In[28]:


    batch = next(iter(TRAINLOADER))


    # In[16]:


    y = model.batch_forward(batch.timeseries)
    torch.linalg.norm(y).backward()
    model.zero_grad()


    # ## Initalize OptimizerDEVICE

    # In[17]:



    # In[18]:


    OPTIMIZER_CONFIG |= {"params": model.parameters()}
    optimizer = initialize_from(OPTIMIZERS, **OPTIMIZER_CONFIG)


    # In[19]:


    # lr_scheduler = initialize_from(
    #     LR_SCHEDULERS, LR_SCHEDULER_CONFIG | {"optimizer": optimizer}
    # )


    # ## Logging Utilities

    # In[20]:



    # def log_all(i, model, writer, optimizer):
    #     kernel = model.system.kernel.clone().detach().cpu()
    #     log_kernel_information(i, writer, kernel, histograms=True)
    #     log_optimizer_state(i, writer, optimizer, histograms=True)


    def log_hparams(i, writer, *, metric_dict, hparam_dict):
        hparam_dict |= {"epoch": i}
        metric_dict = add_prefix(metric_dict, "hparam")
        writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)


    # In[21]:


    RUN_START = tsdm.util.now()
    CHECKPOINTDIR = Path(
        f"checkpoints/{MODEL.__name__}/{DATASET.name}/{RUN_NAME}/{RUN_START}"
    )
    CHECKPOINTDIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR = f"runs/{MODEL.__name__}/{DATASET.name}/{RUN_NAME}/{RUN_START}"
    writer = SummaryWriter(LOGGING_DIR)


    # # Training

    # In[22]:


    batch = next(iter(TRAINLOADER));


    # In[23]:


    yhat = model.batch_forward(batch.timeseries)
    y = batch.encoded_targets
    LOSS(y, yhat)


    # In[24]:


    len(batch.timeseries)


    # In[25]:


    @torch.no_grad()
    def get_total_loss(dataloader):
        ys = []
        yhats = []
        for batch in tqdm(dataloader):
            # ts = batch.timeseries
            # inputs = [(t.to(device=DEVICE),v.to(device=DEVICE), m.to(device=DEVICE)) for t,v,m in ts]
            # yhats.append(model.batch_forward(inputs))
            yhats.append(model.batch_forward(batch.timeseries) )
            ys.append(batch.encoded_targets.to(device=DEVICE))
        return torch.cat(ys), torch.cat(yhats)


    # In[26]:


    ys, yhats = get_total_loss(TRAINLOADER)


    # In[ ]:


    dataloader = EVALLOADERS[key]
    for batch in tqdm(dataloader):
        yhat = model.batch_forward(batch.timeseries)


    # In[ ]:


    i = -1
    epoch = 1

    with torch.no_grad():
        # log optimizer state first !!!
        # log_optimizer_state(epoch, writer, optimizer, histograms=True)

        for key in ((0, "train"), (0, "test")):
            dataloader = EVALLOADERS[key]
            y, ŷ = get_all_preds(model, dataloader)
            assert torch.isfinite(y).all()
            log_metrics(
                epoch, writer, metrics=metrics, targets=y, predics=ŷ, prefix=key[1]
            )


    # In[ ]:


    for g in optimizer.param_groups:
        g["lr"] = 0.0001


    # In[ ]:


    for epoch in (epochs := trange(epoch, 100)):
        batching_time = perf_counter()
        for batch in (batches := tqdm(TRAINLOADER, leave=False)):
            batching_time = perf_counter() - batching_time
            i += 1
            # Optimization step
            model.zero_grad(set_to_none=True)
            times, inputs, targets, originals = prep_batch(batch)

            forward_time = perf_counter()
            outputs = model(times, inputs)
            forward_time = perf_counter() - forward_time
            predics = outputs[:, OBS_HORIZON:, task.targets.index]
            mask = torch.isnan(targets)
            targets[mask] = torch.tensor(0.0)
            predics[mask] = torch.tensor(0.0)
            mask = torch.isnan(originals)
            originals[mask] = torch.tensor(0.0)
            outputs[mask] = torch.tensor(0.0)

            if not FORECAST_ALL:
                # get rid of nan-values in the targets.
                loss = LOSS(targets, predics)
            else:
                loss = LOSS(originals, outputs)

            # # compensate NaN-Value with upscaling
            # scale = 1/torch.mean(mask.to(dtype=torch.float32))
            # targets *= scale
            # predics *= scale

            backward_time = perf_counter()
            loss.backward()
            backward_time = perf_counter() - backward_time

            optimizer.step()

            # batch logging
            with torch.no_grad():
                logging_time = time()
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
            torch.jit.save(model, CHECKPOINTDIR.joinpath(f"{MODEL.__name__}-{epoch}"))
            torch.save(
                {
                    "optimizer": optimizer,
                    "epoch": epoch,
                    "batch": i,
                },
                CHECKPOINTDIR.joinpath(f"{optimizer.__class__.__name__}-{epoch}"),
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


    get_ipython().run_line_magic('matplotlib', 'inline')
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

    S = np.arange(len(xhat_pre_mean));


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
    plt.hist(chg, bins="auto");


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




if __name__ == "__main__":
    run()
    
# %%
