r"""Tasks make running experiments easy & reproducible.

Task = Dataset + Evaluation Protocol

For simplicity, the evaluation protocol is, in this iteration, restricted to a test metric,
and a test_loader object.

We decided to use a dataloader instead of, say, a key to cater to the question of
forecasting horizons.

Decomposable METRICS
--------------------

**Example:** Mean Square Error (MSE)

.. code-block:: python

    dims = (1, ...)  # sum over all axes except batch dimension
    # y, yhat are of shape (B, ...)
    test_metric = lambda y, yhat: torch.sum((y - yhat) ** 2, dim=dims)
    accumulation = torch.mean

**Recipe**

.. code-block:: python

    r = []
    for x, y in dataloader:
        r.append(test_metric(y, model(x)))

    score = accumulation(torch.concat(r, dim=BATCHDIM))

Non-decomposable METRICS
------------------------

**Example:** Area Under the Receiver Operating Characteristic Curve (AUROC)

test_metric = torch.AUROC()   # expects two tensors of shape (N, ...) or (N, C, ...)

.. code-block:: python

   score = test_metric([(y, model(x)) for x, y in test_loader])
   # accumulation = None or identity function (tbd.)

**Recipe**

.. code-block:: python

    ys, yhats = []
    for x, y in dataloader:
        ys.append(y)
        yhats.append(model(x))

    ys = torch.concat(ys, dim=BATCHDIM)
    yhats = torch.concat(yhats, dim=BATCHDIM)
    score = test_metric(ys, yhats)

Normal Encoder
--------------

A normal pre_encoder is a pre_encoder with the property that all output tensors
share the same index axis.

I.e. it has a signature of the form ``list[tensor[n, ...]] -> list[tensor[n, ...]]``.
Pre-Encoder: Map DataFrame to torch.util.data.Dataset



Default DataLoader Creation
---------------------------

.. code-block:: python

    data = pre_processor.encode(data)  # DataFrame to DataFrame
    data = pre_encoder.encode(data)  # DataFrame to DataSet
    dataset = TensorDataset(*inputs, targets)
    sampler = SequenceSampler(tuple[TimeTensor], tuple[StaticTensor])
    dataloader = DataLoader(dataset, sampler=sampler, collate=...)
    batch = next(dataloader)



The whole responsitbility of this class is to use a key from the "sampler" and
create a `Sample` from the `TimeSeriesCollection`.





                                       ┌─────────────┐      ┌─────┐
                                  ┌───►│FoldGenerator├─────►│Folds│
                                  │    └─────────────┘      └──┬──┘
          ┌────────────────────┐  │                            │
          │TimeSeriesCollection├──┤           ┌────────────────┘
          └───────────────┬────┘  │           ▼
                          │       │    ┌──────────────┐     ┌──────┐ fit  ┌───────┐
                          │       └───►│SplitGenerator├────►│Splits├─────►│encoder│
                          │            └──────────────┘     └───┬──┘      └──┬────┘
                          └─────────┐                           │            │
                                    │                           │            │
              ┌─────────────────────┼──────────┬────────────────┘            │
              │                     │          │                             │???
              │        ┌────┐       │???       │???    ┌───────┐             │
              ▼     ┌─►│key1├─┐     ▼          ▼    ┌─►│sample1├──┐          ▼      ┌──────┐
          ┌───────┐ │  └────┘ │  ┌───────────────┐  │  └───────┘  │  ┌──────────┐   │batch │
          │sampler├─┤         ├─►│SampleGenerator├──┤             ├─►│collate_fn├──►│Tensor│
          └───────┘ │  ┌────┐ │  └───────────────┘  │  ┌───────┐  │  └──────────┘   │CPU   │
                    └─►│keyN├─┘                     └─►│sampleN├──┘                 └───┬──┘
                       └────┘                          └───────┘                        │
                                                                                        │
                                        (copy tensors to GPU)                           │
              ┌─────────────────────────────────────────────────────────────────────────┘
              │
              │                           ┌───────┐                                ┌─────────┐
          ┌───┴──┐  ┌────────────────────►│targets├─────────┬───┐              ┌──►│targets  ├─────┐
          │batch │  │                     └───────┘         │   │   ┌───────┐  │   └─────────┘     │  ┌───────────┐
          │Tensor├──┤                                       │   ├──►│decoder├──┤                   ├─►│test_metric│
          │GPU   │  │  ┌──────┐  ┌─────┐  ┌───────────┐     │   │   └───────┘  │   ┌───────────┐   │  └───────────┘
          └──────┘  └─►│inputs├─►│model├─►│predictions├─┬───┼───┘              └──►│predictions├───┘
                       └──────┘  └─────┘  └───────────┘ │   │                      └───────────┘
                                    ▲                   │   │
                                    │                   │   │
                                    │                   ▼   ▼
                                    │   ┌─────────┐    ┌─────┐
                                    └───┤optimizer│◄───┤loss │
                                        └─────────┘    └─────┘

https://asciiflow.com/#/share/eJzdWLtu2zAU%2FRWBUwtkcbu0XjwEqLcgQJJNQKDKTEtUL0gMEDcIYBgdO3AQXA0dM2bqWOhr9CWVKZEixYeoeKs
gxBLJe3nv4bmHYh5BEsQQLJP7KDoDUbCFOViCRx88%2BGD58f3izAfb9undh%2BMThg%2B4ffGBZ7ua8mdT7txuYjYhvp84TXH425T7T2m0WcME5gFO86b8
PXbHBxXtr9XzvvspnXOomCU3eeFd7Uxz4DBh1EdlCZpOtL9GMbyCOYLFeRpFMMQoTQQwnl%2B5SkOqdBp3aEZwcLzkfJpDrVmRYQxfk1kRE1uaxLtD2OZSx
z4xIgEDSq2rLELYSMBhSGHiJkzCdANzPTtH6z9zCSojP3cKe19Ua208tTUGYmKqLhdbOU5R3%2Bjs5Krb0RxnEdt9LTTh6rOU2idU4Ni%2FWq1UjwoWRDJQ
XixVMZ6PTXaoxZkoob%2FB7YKRva%2FFfpT4IloUQZxFcCFUCFHGmyvaKrWkw0MhbcXa57KFibK5DCrP2TfpwPwc4PCrx8W8g4NrySDefUsnKnSQTnhEsZd
twnZvCDC8vRP2BtpxDZMizVkA5sT0eZkwdqmJCaSm0RadHceeX95IDGUolCI%2FL1j%2BUlVqR3eLcSHgq9rI0XH1UARUw0LjZeWXyUS3nZ52ufh8E6bZ1s
OUQ4WHU299efPWyScF63%2B%2BhzyHJysybgI8y7R3oMhkLfg%2BddekpYKD%2FAvUfuhIt7h5ElMSokMuYpqEBu207aXTpdTZsr9TcjSpR9Xg1A3aPhsmx
AYxty1%2FF5Yk6xvYf1pa3Yk7BIYFvo0hzlHIt6M1U1R7QgYWOZOLSHnYQa6ml2rsesYWZTnsVPLmgJLsnhOetsQt4JHYkOVwg%2Bh5jBeGWAC17FxPW5Oj
IVwXvlf6HkOzzl5cnFl1NkbXwuPDHwu7X3NiOMmWfnvqj6nqDBMC7BkG2f7fMQb1Oc0witF3el5sfv0Qu6K0KCbzVF1qiWKgig%2BewNM%2FqIfMMw%3D%3
D)


▄▄▄▄▄▄▄ ▄   ▄▄    ▄▄▄▄  ▄  ▄ ▄▄ ▄ ▄ ▄▄▄▄  ▄▄▄▄ ▄▄ ▄▄▄  ▄ ▄ ▄▄▄  ▄▄ ▄▄▄▄ ▄ ▄▄▄     ▄  ▄▄▄   ▄▄▄      ▄ ▄▄▄▄▄▄▄
█ ▄▄▄ █ █ █▀█  ▀  ▄█▀ ▄ ▀ █▀▄▀█▄▀█ ▀ ████ █▄▀▀▄ ▀ █ ▄ █▄ ▀ ▀▀ ▀ ▄▀▀▄ ▀▄▀  █ ▄▀▀█ ▀ ▀▀ ▀ ▄█▀█▀█▄  ▀█▄█ █ ▄▄▄ █
█ ███ █ ▀▀██  ▀▀▀ ▄█▄ ▄█▀▀  ██▄██▄▀▀ ▄█▄█▀ █▄▀▄   ▄▄▄█▄█▄▀▄▄▄█ ███ █  ▀ ▄ ▄ █▄▄▄▄ ▀ █ ▄█▀▄  ▄██▀ ███▄ █ ███ █
█▄▄▄▄▄█ █▀▄ █▀█▀█ █▀█▀▄▀▄ █ █ ▄ █ ▄ █ █ ▄▀█▀▄▀▄ █ █ █ ▄ █▀▄ █ █▀▄▀█ ▄▀▄▀▄ █▀█ ▄ █▀▄ ▄ █▀█▀█▀█▀▄ ▄ ▄▀▄ █▄▄▄▄▄█
▄▄  ▄▄▄  ▄ █ ▄█▄▄█▄ ▄ ▄  ▄▄ █▄▄▄█   ▀ ▀▄█▄▄▀▄█     ██▄▄▄█ ▀▄ ▄▄▄▀▀▀ ▄█▀█▄ ▄ █▄▄▄██▀▄▄█▄▀███▀▄ █▄ ▀█▀   ▄ ▄▄▄▄
▀▄▄▄█▄▄████▄██▀█▀▄▀█ ▄ ███ █ ▀ ▄▄█▄█ █   █▀█  ▄█▄█   ▀▀▄▀▀▄▄▀▄ ▄▄ ▀ █ ▄▀▀█ ▀ ▀ ▄█ ██▀▄ ▄▀▀ ▄█ ▄▀▄ ▀▄█ █   █▀
█ ▄   ▄█▀▀█▀█ ▄█ █ █▀██▄▄  █▄▀ █▄     █ █▄▀▀  █   ▄▄▀▀█▀▄█▀▄▄ █▄████▄▀   ▀▄ █▀▀▄▄▀▄▄▄ █▄▀█▄▄▄▀█▄▄█▀█▀▄▄ █ █▄█
▄▄▀ ▄▄▄▀▄▄▀▀▄▀▄█▀▀ ▀ ▄ ▀█▄    ▀  ▀█ ▄█ ▀   ▄▀▄▄ █▄ ▀   ▀  ▄█▀█▀▀▄ ▀█▄▀▄ ▀ ▀▀   █  █  ▄ █   ▀█ ▄▀█▄▀██▀█▄  ▄▀▀
█▄ ▀▄▄▄█ ██▀▀▀▀▀▄▄ █  ▄█ ██▄ ▄▀▀▄▄▀ ▄▀▄██ ▀█ ▄▄   █▄ ███▄▀   ▄█▄██▀▀ █▄ ▄▀▀▄▀█▄█ ▀▀ ▄█▄ ██    ▄▄ ▀▄▄ ▄ ▄▄ ▀▀█
█ ▀▀▄▄▄█▀▄ ▄▄█▀▀▀████▀▄▀ ▀█▄ ▀  █▄▄  ▀▀ ▀▀█▄ ▄▄ ▄▀▀▄ ▀▄█▄█▄▀▄▀▀█▀▀▀  ▄█▀▄▀▀▄█▀ █ ██▀▄▀     ▄▀▄█▄▄  █▄▄▄▀▄█▀▀
▀█▄▀▄▄▄██▀ ▀█▀▄▀█▄▀█▀ ▄ ▀█ ▀ █▀█    ▄██▀█ ▄█▄▀  ▄▄ █▀▀▀█     ▀█▄▀▀▄█ █▀▄▄ █▀ ▄▀▀ █ ▄  ▄ █ ██ █▀  ███▀▀ ▄ ▄█▀▄
▄▄▀▄▀▄▄ ███▄  ▀▀▀█▀▄▀█ █▄▄▀█▀ ▄▄▀▀█  ▄▀▄▀▀ ▀ ▀█ ██▀▀▄█▀▀▀ █ █ ▀ ▄▀▀ █ ▄▄ ▄▀▀ ▀ ▀▀▀█▀▀ ▀▀▀█ ▄████▄▄ ▀ █▄█▀▄  ▀
▀█▀█▄ ▄▄█▄▄▀▀▄█▀▀█▄██ ▀▄ ▄▀  ▀  ▄▀▄  ▄ ▄▀▄██▄ ▄  ▀▄██▄██ █ ▀ █▀▄▀▀▄█ ▄▀   ▀▄█ ██▄▀▀▄ ▄█▀███▄▄▀  ▄▄▀█ ▄ ▄█ ▀▄▄
█ ▀█▀█▄▄   ▀ █▄▄██  █▀  █▀▀▀ ▀ ▀█▀▄▀▄█▀█▄▀ ▄▀▄▄ ▀█ █▄█▀█▀ ▄ ▀▄ █▀  ▄ ▀█ ▄▄▀▄▄▀ ▀█▀██▄█   ▀  █▀█▄▀█▀██    ▄  ▀
▀▄█▀▄▄▄█▄▀▄█▄▄█▄▀ ██ █▀█ ███▄▄▄█▄ ▀▄ ▀▄ ▀█▄▀ ▄▄ ▄▀█▀█▄██▄▀█  ▄▄ ▀▀██▄▀ ▄ ▀█ ██▄▄▄▄▀ ▄ ▄▀██▄█ ▄█▄▄█▄▄▄▄▄▄█▄█▀
▄ ▀█ ▄ █ ██▀█▄▀██ ▀▀▀ ▄▄▄ ██ ▄ ██▄█▀▀▀▀   ▄▄█▄▀▄ ▀▀█ ▄ █ ▄▀▄▀ ▀▀  ██ █▀█▄▀ █ ▄ █ ▄█▄█▀▄   ██▀▄ ▄▄ ▄█ ▄ ██▄ ▀
▀▀▀██▄▄▄██▀▄█▄██▀ ▀ █▄▀▄▄█▀██▄▄▄█ ▀  ▄█▄▀▀▄█▄▀  ▄▀▀▀█▄▄▄█▀▄  ▀▀█▀█▄▀▄█▀▄▄▄▀▀█▄▄▄█▀ ▄▄▄ ▄█▀█▀ ▀█▄▄▀▀▀█▄▄▄█▄▀▀
▄▀▀▀▄▄▄▄  ▄▀█▀█    ▄▄  ▄ █▀▄▀▀▀█ ▄█ ██   ▀▀▄█ █▀▄▄▀ ▄▄▄▀ ▀▄▄█▄▀█▄▄▀▄█ █ ▄▄ █ ▀▀▀█▀█ ▀▀▀  ▀▀█▄▀▄▀▀▄▀█ ▀█▀██▄
██▄▄ █▄▄ █▄▀█▄██▀▄ ▀ ▄  ▄ ▀▀ █ ▀█▄▄▀▄▄█▀▀  █ ▀▄ ▄ ▄ ▄ ▄▀█▄ ▄ ▀ ▄█▄██  █▄▄   ▀ ▄▀█▀▄▄▄█▀▀▀█▄ ▄▀█▄  ▄█▀█▄▀ ▄▀ ▄
▄█ █▄▀▄▄▀▄▄ █▀   ▄▀▄▀▄█▀▀█ ▄▀█▀ ██▄██▀ ▄█▄▀ ▀▀▄  █▀█▄▀█ ▄ ▄▀▄█▀█     ▀▄█▀▀▀█▄  ▀  ▄▄▀▄▀▀    █▀██▀ ▀ ▄▄███▄▄█
█ ▀▄▄ ▄▄▀   ▄▄▄▄   ▄   ▄▄█    ▄▀█  ▄ ▀█ ▀▄█▀ ▀▄▄▄▀▄   ▄▀█▄ ▄▄█▀▄▀▄▄█▄▄▀  ▀▀  █ ██▀█  ▄█ ▀█▄▀ ▄▀▄ █▀█▀██  ███
█▄ ▄ ▄█▀▀▀▄▀██▀ ▄▀█ ███ █▀▄▀▀█ ▀▀▄█ █ ▀   ▀▀▀█▄ ▄▀ ▄█▀▄▄ █▀   ▀   ▀▄▄███▄ █▀█▀█  ▄▄ ▄ ▀  ▀▀█▄▄███   ▄▄██ █▀▀
▄▄▄▄ ▄ ▄▄▄ ███▀▄ ▄█ ▄▀▄▀ ▄▄ ▄  █▄▄▀█▀▀█▀▄█ ▄ ▀▄  ▄▄▀▄▀█▄ ▀█ █ ▀▄  ▄█▄▄█▀    ██ ▄▄ ▀ ▀ ▀▀▄ ▄▄ █▄▀▄█▄█▄▄▀██
█ █▀▀▀▄███▀ ▄▀▀▄██▄▀▀▀▄  █ ▀  █▀█▀▄▄▄ ▀█▄  ▀▀ ▄█▀▄▀▄▄▀█▀ ▀█ █▄ ▀   ▀▄▀█ ▄█ ▄▄  ▀███▄█▄  ██▀▄▀ ██ █ ▀ ▄▀█▄█ ▀▀
██▀▄█▀▄▄ █▄█ █ ██  ██ ▀ ▄▄ ▀▄▄▄ █ ▀▄  █▄█▀█▀▄▀▄  ▄█ ▄▄  ▄█▀█ █ █▀ ▀▀▄█ ▄ █ ▀ ▄▄ ██▄  ▄▄ ▀▄▀█  ▄  ▄ █▄ █ ▄██▀▄
▄▀▄███▄ ▄  ▀ ▄███▄▀▀█▀█ ██ ▀█▀▄▄ ▀▄█▄▄▀█    █ ▄ ▀   ▄█▀█▀ █ ▄█▀█▄▄▀   █ ▄█▀▄███▀ ▀███▄▀▀ █▀▀▄ █▄█▄▀█▀▄▄███▀▀▀
▀▄▄▄█▄█▄▀▀ ▀▀▀▀▄█▄ ▄█▀▄ ▀▀▀▄▄▄▄▄▄▀ ▄██ ▀▄█▀  █ ▄██▀▄▄▄▄██ █▄ █ ██▀▀▄█  ▄▄█▀▄▄▄█▄▀█   █▄▀▄█ ▄█▄ ▄ ▄▀▄▄▄▄▄ ███
█▄█▀█ ▄ █ ▄▄▄█▀▄ █ █▀█▄  █▄ █ ▄ █ █ ██  █▀▀█▀ █▀█▄▀▀█ ▄ █▀▄▀█▄  ▄▀▀   █  █ ▄█ ▄ █▀█ ▄█▀▄    ▄▀██  ▀ █ ▄ ██▄▀▀
▀▀█▀█▄▄▄█   ▀██▀▄▀██▀▀██▄ ▄ █▄▄▄██▄▀▄██ █ ▄▀  ▀█ ▀▄██▄▄▄█ ██▄ █ ▀▄ █▄█ ▄ ████▄▄▄████    █▄▀▀ ██▄ █▀▄█▄▄▄█▄███
▄█▀▀▀ ▄█▀▄ █▄█▀██ ▀██▄█▄█▄▀▀█▀▀ ▀▀▄▄▄ ▀█▀ ▀▄█▄█▀██▀ █▄▀▄▀ ██▀▄▀█▄ ▀▀█▀▄▀▄ ▀██▄▀▄▀ ▄▄▀▄ ▀▄▀ ▄█ █ █▄▀█ ███ ██ ▀
▀▄█ █▄▄█ ▀▀▄ ▀██ ▀ ▄▄ ▄▄ █ ██▄▄█▄ ▄ ██ ▀ ▄  ▄ ▄ ▀█▄ ▀█      ▀▄ █▀█▀▄▀█▄▄█▀ ▀▄▄█▄  ▄ ▄█▄▀ ▀▀▄▀ █ ▄▀ █▄▄▀█ ▀▄█
▀▀▀██▀▄  █ █ ▀▄█▀▄█▀█ ▄███ █▄▀ ▄█ █  █▀▀ ▀▀ ▀▀▄▄▀▄ ██▀ ▀▀▀█▄▀█ ▄▄▀ ▄█▀█ ▄▄▀ ▄█▀▄█ ██▀▄  ▄▀ ▀▄▀▄▀▄▄▀█ ▀▀█▀▄█ ▀
▄▄    ▄██▀▀▄ █ █▄▀▀█▀  ▄▄▄▀ ▀▀█ ▄▄▄  ▀▄█▀█ █ █ ▄▄▄█▀████▄▄   ▄█▄▀ ▄ ▄▄ ▄▄▀▀ ███  ▄▄▄  ▄▄██▄▀▄ ▀  ▄███▄▀▀█▄▀▀█
▄█▀▄█▀▄  █▄  █▀▀ ▄▄▀▄█▄ ▄█▀ ██ ▄█ ▀▀▀ ▀▀▄▀▀▀▀█▄▄ █ ▀▄█ ▄▀▀█▄▄▄     ▄█ ███▄  ▄ ▀▀▀ ▄▄▄█▀█▄▄▀▀▄▀█ ▄█▀▀▄▄█▄▄  ▀
█▀███▄▄ █▄ ▀█ ███▀▄▀▀▄ █▄█▀▀█ ▄█▄▀█▄ ██▄▀▀ █▄▄ ▄  ▄▄ █▄▀   ▄ ▄█▄▀▄▄▄▄█▀▄  █▄ █▄  ██   ▄ █▄▀█  ██▄▄█▀▀  ▀█▄█▄█
▄█▄   ▄ ▄▀▀▀▀▀ ▀█▀▀▄▄  ▄ ▄▀▄███ ▄▀▄  █ ▄█ ▀██▀█ █▄▀▀██▀▄▀ ▄▄▀▄ ▄ ▀  ▄▀▄██▀▀██▄█ █▀█ ▀█  █  ▀▄ █ ██ █▄ █▀ ▄██
█ ▄ █ ▄█  ▄▀█▀██▀▀▄▀██ ▄ █▀ ▀██▀▀▀▀▄ ▀█ ▀█▄█▄▀▄ ▄ ▄▀ ▀█ ▀   ▄▄█▄▀▀██▄█ ▄ ▄█▄▀▄█ ▀█ ▄▄▀▀ ▀▄█ ▄▀ ▀ ▄█ ▀▀█▀█ ██▄
▄▄▄▀▀█▄ ▄ █▄▀█▄ ▄▀▄▀ ▄▄▄▀█ ██ ▀ ▀ ▄█▀▄▀▄▀ ▀█▄ ██▀▀▀███▄▀▀▀█▄ ▄ ██ ▀ █▀█▄▀█  █▄ ▀▀ █▀██▀ ▄  ▀▀ █▄▀█ ▀ ▀██▀▄
█▄█ ▄█▄██▄▀ ▄  █▄█▀██▄▀  ▀█ █▄▄▄█▄▀▄ █  ▀▄   ▄▀▄ █▀▀██▄▄▄▄▀▄ ▄█▄▀█▄▀▄ ▀  ██ █▄█▄█▀ ▄▄ ▀ ▀▀██   ▄  ▄ █▄▄██ ▀██
█ █ █ ▄ █▄  █ ▄ █▄▄▄ ▀█▀█ ▀ █ ▄ █▀▄██  ▄ ▀▀▄▄ █ ██  █ ▄ █▄█▄▄█▀▀▄▄  █▀█▄▀   █ ▄ █▀█ ▄ ▀  ▀▀▀▄ ▄▄▀█  █ ▄ █  ▀
█▄▀▀█▄▄▄█ ▀█▄ █ ▀▄▀█▀▀  ▄▀█▄█▄▄▄██ ▀▄▀███  █ ▀█▄▄▀█ █▄▄▄█▄  ▄▄▄ ▀▄█▀▄▀ ▄ ▀█ █▄▄▄██ ▀▄▀▀▄▀▀▀▀ ▀▄ ▄▄█▀█▄▄▄█ █▀
█▀▀▀  ▄▀▀ █▀▀█▀▄  ██ ▀ ▀▀▀ ▄ ██▄▀▀▄█▀ ▀▀ ▀▀ █▄▄▀██▀▄█▀▀█▀ ▄█ ▄▀▀▀▀ ▀▀▀█▄▀▄ ▄██▀  █▄  ▄▀█▄ ▀▀▀▀▄▄ █ █▄▀▀█▀ ▄▀▀
▀▄ █ ▄▄▀▄ ▀ ▄ ██▄▀█ ▀█  ▄█▀▀█ ▄█▄█ ▄▄███▀▀▀█ ▄ █▄▀██ █▀▀▄█▀   ▄█▀█▄█▄▀ ▄▄ ▄█▄▄▄▀▄▄▄▄  █▄█ ██▄▄▄▄▄ █▀ ▀ ▀ ▄███
▀▀▀█▄█▄▄▀█     ▄▄▄██▀ ▄▄ █ ▄  █   ▄▀▀▄  ▄▀▀▀███▀▀█▀▄ ███▀ █▄█  ▄ █▀▀█▄█ █▀  ██▀█  ▄▀▄▄ █▀▀▀ ▄ ▄███ ▀▄▀  ▀ ▀▀
▄ █ ▄▄  ▄ ▄  █▀▄██ ███▄▄▄█▀█▀ ▄▀▀ ▄▄ █▄▀ ▄█▄ ▄ ▄█▄▄█▄▄█▀▀▀█  ▀ ▀ █▀▄▄ ▀▄█▀ █▄▄▄█▄▀  █▄ ▀ ▄▀ ██  ▀█ ▄██ ▄ █▀
██▀  ▀▄█ ▄▄ █ ▀█  ▀  ▀▄█▀ ▀ ▄█▀█  █▄█▄▀▄▀█ ▄▀ █▀██  ▀█▀█▀▀█▄██ ▀▄▀▀█▄▀▄▄▀▄▀▄████▀██▄ █  ▀▀▀▄ ▀▄▀█▄ ▀▀ ▀ ██▄█
▀▄█  █▄██▄ █ ▀▀█  █▄█▀▄ ▀▀ ▀█▀▀ ▀ █  ▄█▄█ █▀▄▀▄▄▄█▄██▄▄▀█▀ ▀▄▄█▄█▀▀█ ▄▀▄ ▄▄ ██▄██▄ ▄▄ ▄▀█▀█▀▄▀ ▀▄█ ▀ ▄▄▄ ▄█▀▀
▀▀ ▀▀▄█▄▄ █▀ ██  ███▀▄▀▄▄  ▀██▄  ▄█▀▄ ▄▄▄  █▀▄█▀█▀▄███▄▀██▀▄▀ ██ ▀██▀█▄█▀   ███▀ ▄▀ ▄▀ ▀▀   ▀█▄▄▄▀ ▀█▀ ▀▄▀
▄█ ▄▄▄▄▀ ▄ ██▄█ ▄▀█▄▄▀▄▀▄▄███▄▄▄▀█▀   █▄█ ██▄ █ ▄ █▄ ▀ █▀  █▄█▀ █▀▀▀▄▄▀▄ ▀██ ▄█▄█ ▄▄▄ ▄█▀███▄▀   ▄▀   ▀▄▄ █
▀██▄█▀▄ ▀█▄█▀▄▀▀█▄▄█▀█▄▀ ▀ ▄▀▀▀▄ ▀██    ▄█ █▄ █ █  ▄▄▀█▄  ██▀▄▀   ▀█▄██ ▀█ ▄▀▀▀█ ▄▄▄▄█ ▀  ▀█▄▀▄▀█▀▀▀▄█ ▄█▄█
▄▄█▀█▀▄▀█▄ ▄▀ █▄▀ ▀▀ ██▄▄█▀▀████▄ ▀  ██ ▀▄▀█ ▀▀▀▄▄█▀██▄█▄ ▀   █▄▀███ ▄▄   ▀█▄▄▄█▄   ▄▄▄█▀██▀ █▀ ▄▀▀ ▄▄█▄▄▄█▀
▄▄▄▄▄▄▄ ▀▄▄ █▀▄▀▀▄▀▄▀  ▀█▀ ▀█ ▄ ██▄▄█▄ ▀▄▀▀▄█ ████ ██ ▄ ██▄▄ ▄▀▀▄▀▀▀█▄▄ ██▀▄█ ▄ █▀█  █ ▄█▀ █▀█▄▄ █ ▄█ ▄ █ ▄
█ ▄▄▄ █ █ ▄▀ ▄█ ▀▀▄  ▀█▀ █▀██▄▄▄█ ▀▄▄▄▄ █ ██ █▀▄  █▀█▄▄▄█  ▄    ▀▀▀▀▄▄  ▄ ▄▀█▄▄▄█ █ ▄▄▀▄▀  ▀▄ ▄▄ █  █▄▄▄███ █
█ ███ █  ▀██▄▄█ ▄   █ ▄▄▀█▀███▀▀█▀█ █▄▀ ▄█▀▀▄ ▄▀██▀ ▀ ▀██▀▄▄▄▄ ▄ ▄▀▄▀ █ █ ▀███▀ ▀ █ ▀▀ ▀▀   █▀█▄ █▀ █▄▀▄  ▄▄▀
█▄▄▄▄▄█ █▄█▄▀▄▄▄██▄▀▄▄▄  ▀▄ █▄█▀ ▄▀▀▄▀▄ ▀▄█▀▄▀█▄  █ ▀ ▄▀▄▀▀   ▀ ▀▄▀█  ▄  ██ █▀█▀  ▀▄ ▀▀ █▄▄█ ▄   ▀▄▄▄ ▄▄█ ██▄
"""

from __future__ import annotations

__all__ = [
    # Classes
    "OldBaseTask",
    "TimeSeriesTask",
    "TimeSeriesSampleGenerator",
]

import logging
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
)

from pandas import NA, DataFrame, Index, MultiIndex, Series
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler as TorchSampler

from tsdm.datasets import Dataset, TimeSeriesCollection, TimeSeriesDataset
from tsdm.encoders import ModularEncoder
from tsdm.utils import LazyDict
from tsdm.utils.strings import repr_dataclass, repr_namedtuple
from tsdm.utils.types import KeyVar

TimeSlice: TypeAlias = Index | slice | list

SampleType_co = TypeVar("SampleType_co", covariant=True)

Batch: TypeAlias = Tensor | Sequence[Tensor] | Mapping[str, Tensor]

TS_Type_co = TypeVar(
    "TS_Type_co", bound=TimeSeriesDataset | TimeSeriesCollection, covariant=True
)

SplitID = TypeVar("SplitID", bound=Hashable)


class BaseTaskMetaClass(ABCMeta):
    r"""Metaclass for BaseTask."""

    def __init__(cls, *args, **kwargs):
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        super().__init__(*args, **kwargs)


class OldBaseTask(ABC, Generic[KeyVar], metaclass=BaseTaskMetaClass):
    r"""Abstract Base Class for Tasks.

    A task is a combination of a dataset and an evaluation protocol (EVP).

    The DataLoader will return batches of data consisting of tuples of the form:
    `(inputs, targets)`. The model will be trained on the inputs, and the targets
    will be used to evaluate the model.
    That is, the model must product an output of the same shape and data type of the targets.

    Attributes
    ----------
    train_batch_size: int, default 32
        Default batch-size used by batchloader.
    eval_batch_size: int, default 128
        Default batch-size used by dataloaders (for evaluation).
    dataset: Dataset
        The attached dataset
    """

    # __slots__ = ()  # https://stackoverflow.com/a/62628857/9318372

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""
    train_batch_size: int = 32
    r"""Default batch size."""
    eval_batch_size: int = 128
    r"""Default batch size when evaluating."""
    preprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific preprocessor (applied before batching)."""
    postprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific postprocessor (applied after batching)."""

    def __init__(self) -> None:
        r"""Initialize."""
        warnings.warn("deprecated, use new class", DeprecationWarning)

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        string = (
            f"{self.__class__.__name__}("
            # f"dataset={self.dataset.name}, "
            f"test_metric={type(self.test_metric).__name__})"
        )
        return string

    @property
    @abstractmethod
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The metric to be used for evaluation."""

    @property
    @abstractmethod
    def dataset(self) -> Dataset | DataFrame:
        r"""Return the cached dataset associated with the task."""

    @property
    @abstractmethod
    def index(self) -> Sequence[KeyVar]:
        r"""List of index."""

    @property
    @abstractmethod
    def splits(self) -> Mapping[KeyVar, Any]:
        r"""Cache dictionary of dataset slices."""

    @abstractmethod
    def make_dataloader(
        self,
        key: KeyVar,
        /,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split.

        Args:
            key: From which part of the dataset to construct the loader.
            dataloader_kwargs: Options to be passed directly to the dataloader such as the generator.

        Returns:
            DataLoader: A DataLoader for the selected key.
        """

    @cached_property
    def dataloaders(self) -> Mapping[Any, DataLoader]:
        r"""Cache dictionary of evaluation-dataloaders."""
        kwargs: dict[Any, Any] = {
            # "key": key,
            "batch_size": self.eval_batch_size,
            "shuffle": False,
            "drop_last": False,
        }

        return LazyDict(
            {key: (self.make_dataloader, kwargs | {"key": key}) for key in self.splits}
        )


class BaseTask(ABC, Generic[KeyVar], metaclass=BaseTaskMetaClass):
    r"""Abstract Base Class for Tasks.

    A task is a combination of a dataset and an evaluation protocol (EVP).

    The DataLoader will return batches of data consisting of tuples of the form:
    `(inputs, targets)`. The model will be trained on the inputs, and the targets
    will be used to evaluate the model.
    That is, the model must product an output of the same shape and data type of the targets.

    Attributes
    ----------
    train_batch_size: int, default 32
        Default batch-size used by batchloader.
    eval_batch_size: int, default 128
        Default batch-size used by dataloaders (for evaluation).
    dataset: Dataset
        The attached dataset
    """

    # ABCs should have slots https://stackoverflow.com/a/62628857/9318372
    __slots__ = ()

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""
    preprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific preprocessor (applied before sampling/batching)."""
    postprocessor: Optional[ModularEncoder] = None
    r"""Optional task specific postprocessor (applied after sampling/batching)."""

    dataloader_config_train = {
        "batch_size": 32,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "collate_fn": lambda x: x,
    }

    dataloader_config_infer = {
        "batch_size": 64,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
        "collate_fn": lambda x: x,
    }

    def __init__(
        self,
        dataset: TorchDataset,
        sampler: TorchSampler,
        dataloader_config_train: dict[str, Any],
        **dataloader_kwargs: Any,
    ):
        r"""Initialize the task object."""
        self.dataloader_config_train |= dataloader_kwargs
        self.dataloader_config_infer |= dataloader_kwargs

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        string = (
            f"{self.__class__.__name__}("
            # f"dataset={self.dataset.name}, "
            f"test_metric={type(self.test_metric).__name__})"
        )
        return string

    @property
    @abstractmethod
    def test_metric(self) -> Callable[..., Tensor]:
        r"""The metric to be used for evaluation."""

    @property
    @abstractmethod
    def dataset(self) -> Dataset | DataFrame:
        r"""Return the cached dataset associated with the task."""

    def split_type(self, key: KeyVar) -> Literal["train", "infer", "unknown"]:
        r"""Return the type of split."""
        train_patterns = ["train", "training"]
        infer_patterns = [
            "test",
            "testing",
            "val",
            "valid",
            "validation",
            "eval",
            "evaluation",
            "infer",
            "inference",
        ]

        if isinstance(key, str):
            if key.lower() in train_patterns:
                return "train"
            if key.lower() in infer_patterns:
                return "infer"
        if isinstance(key, Sequence):
            patterns = {self.split_type(k) for k in key}
            if patterns <= {"train", "unknown"}:
                return "train"
            if patterns <= {"infer", "unknown"}:
                return "infer"
            if patterns == {"unknown"}:
                return "unknown"
            raise ValueError(f"{key=} contains both train and infer splits.")
        return "unknown"

    @property
    @abstractmethod
    def index(self) -> Sequence[KeyVar]:
        r"""List of index."""

    @abstractmethod
    def make_split(self, key: KeyVar, /) -> TorchDataset:
        r"""Return the split associated with the given key."""

    @abstractmethod
    def make_sampler(self, key: KeyVar, /) -> TorchSampler:
        r"""Create sampler for the given split."""

    def make_dataloader(
        self,
        key: KeyVar,
        /,
        **dataloader_kwargs: Any,
    ) -> DataLoader:
        r"""Return a DataLoader object for the specified split."""
        dataset = self.splits[key]
        sampler = self.samplers[key]

        match self.split_type(key):
            case "train":
                kwargs = self.dataloader_configs["train"]
            case "infer":
                kwargs = self.dataloader_configs["train"]
            case _:
                raise ValueError(f"Unknown split type: {self.split_type(key)=}")

        kwargs = kwargs | dataloader_kwargs
        dataloader = DataLoader(dataset, sampler=sampler, **kwargs)
        return dataloader

    @cached_property
    def dataloader_configs(self) -> dict[str, dict[str, Any]]:
        r"""Return dataloader configuration."""
        return {
            "train": self.dataloader_config_train,
            "eval": self.dataloader_config_infer,
        }

    @cached_property
    def splits(self) -> Mapping[KeyVar, TorchDataset]:
        r"""Cache dictionary of dataset splits."""
        return LazyDict({k: self.make_split for k in self.splits})

    @cached_property
    def samplers(self) -> Mapping[KeyVar, TorchSampler]:
        r"""Return a dictionary of samplers for each split."""
        return LazyDict({k: self.make_sampler for k in self.splits})

    @cached_property
    def dataloaders(self) -> Mapping[KeyVar, DataLoader]:
        r"""Cache dictionary of evaluation-dataloaders."""
        return LazyDict({k: self.make_dataloader for k in self.splits})


@dataclass
class TimeSeriesTask(
    Mapping[SplitID, DataLoader[SampleType_co]],
    Generic[SplitID, TS_Type_co, SampleType_co],
    metaclass=BaseTaskMetaClass,
):
    r"""Abstract Base Class for Tasks.

    A task has the following responsibilities:

    - Provide a dataset
        - All splits should be derived from the same dataset
    - Provide splits of the dataset, usually one of the following:
        - train, test
        - train, validation, test
        - cross-validation + train/(valid)/test: (0, train), (0, test), ..., (n, train), (n, test)
        - nested cross-validation: (0, 0), (0, 1), ..., (0, n), (1, 0), ..., (n, n)
    - Provide a metric for evaluation
        - must be of the form `Callable[[target, prediction], Sclar]`
    - Provide a `torch.utils.data.DataLoader` for each split
        - Provide a `torch.utils.data.Dataset` for each split
            - Dataset should return `Sample` objects providing `Sample.Inputs` and `Sample.Targets`.
        - Provide a `torch.utils.data.Sampler` for each split
            - Sampler should return indices into the datase
    - Optional: Task specific encoder/decoder
        - Since we use raw format, such an encoder might be necessary to even meaningfully
          compute the target metric
        - For examples, tasks may ask for a MSE-loss on standardized data, or a cross-entropy for
          categorical data. In such cases, the task should provide an encoder/decoder to convert the
          data to the desired format.
        - For the simple case when both the input and the target are DataFrames, the task should
          provide DataFrameEncoders, one for each of the index / timeseries / metadata.
        - Encoders must be fit on the training data, but also transform the test data.
          - Need a way to associate a train split to each test/valid split.
          - ASSUMPTION: the split key is of the form `*fold, partition`, where `partition` is one of
            `train`, `valid`, `test`, and `fold` is an integer or tuple of integers.

    To make this simpler, we here first consider the `Mapping` interface, i.e.
    Samplers are all of fixed sized. and the dataset is a `Mapping` type.
    We do not support `torch.utils.data.IterableDataset`, as this point.

    Note:
        - `torch.utils.data.Dataset[T_co]` implements
            - `__getitem__(self, index) -> T_co`
            - `__add__(self, other: Dataset[T_co]) -> ConcatDataset[T_co]`.
        - `torch.utils.data.Sampler[T_co]`
            - `__iter__(self) -> Iterator[T_co]`.
        - `torch.utils.data.DataLoader[T_co]`
            - attribute dataset: Dataset[T_co]
            - `__iter__(self) -> Iterator[Any]`
            - `__len__(self) -> int`.
    """

    # ABCs should have slots https://stackoverflow.com/a/62628857/9318372
    # __slots__ = ()

    LOGGER: ClassVar[logging.Logger]
    r"""Class specific logger instance."""

    dataset: TS_Type_co
    r"""Dataset from which the splits are constructed."""

    _: KW_ONLY = NotImplemented
    index: Sequence[SplitID] = NotImplemented
    r"""List of index."""
    dataloaders: Mapping[SplitID, DataLoader[SampleType_co]] = NotImplemented
    r"""Dictionary holding `DataLoader` associated with each key."""
    encoders: Mapping[SplitID, ModularEncoder] = NotImplemented
    r"""Dictionary holding `Encoder` associated with each key."""
    folds: DataFrame = NotImplemented
    r"""Dictionary holding `Fold` associated with each key (index for split)."""
    samplers: Mapping[SplitID, TorchSampler] = NotImplemented
    r"""Dictionary holding `Sampler` associated with each key."""
    generators: Mapping[SplitID, TorchDataset] = NotImplemented
    r"""Dictionary holding `torch.utils.data.Dataset` associated with each key."""
    splits: Mapping[SplitID, TS_Type_co] = NotImplemented
    r"""Dictionary holding sampler associated with each key."""
    test_metric: Callable[[Any, Any], Any] = NotImplemented
    r"""Metric used for evaluation."""
    collate_fn: Callable[[list[SampleType_co]], Batch] = NotImplemented
    r"""Collate function used to create batches from samples."""

    train_patterns: Sequence[str] = ("train", "training")
    r"""List of patterns to match for training splits."""
    infer_patterns: Sequence[str] = ("test", "testing", "val", "valid", "validation")
    r"""List of patterns to match for infererence splits."""

    def __post_init__(self) -> None:
        r"""Initialize the task object."""
        if self.folds is NotImplemented:
            self.LOGGER.info("No folds provided. Creating them.")
            self.folds = self.make_folds()

        # check the folds for consistency
        self.validate_folds()

        if self.index is NotImplemented:
            self.index = self.folds.columns
        if self.splits is NotImplemented:
            self.LOGGER.info("No splits provided. Creating them.")
            self.splits = LazyDict({key: self.make_split for key in self})
        if self.dataloaders is NotImplemented:
            self.LOGGER.info("No DataLoaders provided. Caching them.")
            self.dataloaders = LazyDict({key: self.make_dataloader for key in self})
        if self.samplers is NotImplemented:
            self.LOGGER.info("No Samplers provided. Caching them.")
            self.samplers = LazyDict({key: self.make_sampler for key in self})
        if self.encoders is NotImplemented:
            self.LOGGER.info("No Encoders provided. Caching them.")
            self.encoders = LazyDict({key: self.make_encoder for key in self})
        if self.generators is NotImplemented:
            self.LOGGER.info("No Generators provided. Caching them.")
            self.generators = LazyDict({key: self.make_generator for key in self})
        if self.test_metric is NotImplemented:
            self.LOGGER.info("No test metric provided. Using default.")
            self.test_metric = self.default_metric  # type: ignore[assignment]
        if self.collate_fn is NotImplemented:
            self.LOGGER.info("No test metric provided. Using default.")
            self.collate_fn = self.default_collate_fn

    def __iter__(self) -> Iterator[SplitID]:
        r"""Iterate over the keys."""
        return iter(self.folds)

    def __len__(self) -> int:
        r"""Return the number of splits."""
        return len(self.folds)

    def __getitem__(self, key: SplitID) -> DataLoader[SampleType_co]:
        r"""Return the dataloader associated with the key."""
        return self.dataloaders[key]

    def __repr__(self) -> str:
        return repr_dataclass(self)

    def fit_encoder(self, key: SplitID, /) -> None:
        r"""Fit the encoder."""
        self.LOGGER.info("Initializing Encoder for key='%s'", key)
        encoder = self.make_encoder(key)
        train_key = self.train_partition[key]
        associated_train_split = self.splits[train_key]
        self.LOGGER.info("Fitting encoder to associated train split '%s'", train_key)
        encoder.fit(associated_train_split)

    @staticmethod
    def default_collate_fn(samples: list[SampleType_co]) -> Batch:
        r"""Return the test metric."""
        return samples  # type: ignore[return-value]

    @staticmethod
    def default_metric(*, targets, predictions):
        r"""Return the test metric."""
        raise NotImplementedError

    def make_split(self, key: SplitID, /) -> TS_Type_co:
        r"""Return the splits associated with the specified key."""
        return self.dataset[self.folds[key]]

    def make_encoder(self, key: SplitID, /) -> ModularEncoder:
        r"""Create the encoder associated with the specified key."""
        raise NotImplementedError

    def make_sampler(self, key: SplitID, /) -> TorchSampler[KeyVar]:
        r"""Create the sampler associated with the specified key."""
        raise NotImplementedError

    def make_generator(self, key: SplitID, /) -> TorchDataset[SampleType_co]:
        r"""Return the splits associated with the specified key."""
        raise NotImplementedError

    def make_folds(self, /) -> DataFrame:
        r"""Return the folds associated with the specified key."""
        raise NotImplementedError

    def make_dataloader(
        self, key: SplitID, /, **dataloader_kwargs: Any
    ) -> DataLoader[SampleType_co]:
        r"""Return the dataloader associated with the specified key."""
        self.LOGGER.info("Creating DataLoader for key=%s", key)
        dataset = self.generators[key]
        sampler = self.samplers[key]
        kwargs = self.dataloader_config[key] | dataloader_kwargs
        dataloader = DataLoader(dataset, sampler=sampler, **kwargs)
        return dataloader

    def split_type(self, key: SplitID) -> Literal["train", "infer", "unknown"]:
        r"""Return the type of split."""
        if isinstance(key, str):
            if key.lower() in self.train_patterns:
                return "train"
            if key.lower() in self.infer_patterns:
                return "infer"
        if isinstance(key, Sequence):
            patterns = {self.split_type(k) for k in key}
            if patterns <= {"train", "unknown"}:
                return "train"
            if patterns <= {"infer", "unknown"}:
                return "infer"
            if patterns == {"unknown"}:
                return "unknown"
            raise ValueError(f"{key=} contains both train and infer splits.")
        return "unknown"

    @cached_property
    def dataloader_config(self) -> dict[SplitID, dict[str, Any]]:
        r"""Return dataloader configuration."""
        return {
            key: {
                "batch_size": 32,
                "drop_last": self.split_type(key) == "train",
                "pin_memory": True,
                "collate_fn": self.collate_fn,
            }
            for key in self
        }

    @cached_property
    def train_partition(self) -> Mapping[SplitID, SplitID]:
        r"""Return the train partition for the given key."""
        if isinstance(self.folds, (Series, DataFrame)):
            split_index = self.folds.T.index
        elif isinstance(self.folds, Mapping):
            split_index = Index(self.folds.keys())
        else:
            raise TypeError(f"Cannot infer train-partition from {type(self.folds)=}")
        if isinstance(split_index, MultiIndex):
            *fold, partition = names = split_index.names

            # Create Frame (fold, partition) -> (fold, partition.lower())
            df = split_index.to_frame()
            mask = df[partition].str.lower().isin(self.train_patterns)

            # create Series (train_key) -> train_key
            train_folds = df[mask].copy()
            train_folds = train_folds.drop(columns=names)
            train_folds["key"] = list(df[mask].index)
            train_folds = train_folds.droplevel(-1)

            # create Series (fold, partition) -> train_key
            df = df.drop(columns=names)
            df = df.join(train_folds, on=fold)
            return df["key"].to_dict()
        if isinstance(split_index, Index):
            mask = split_index.str.lower().isin(self.train_patterns)
            value = split_index[mask]
            s = Series(value.item(), index=split_index)
            return s.to_dict()
        raise RuntimeError("Supposed to be unreachable")

    def validate_folds(self) -> None:
        r"""Makes sure all keys are correct format `str` or `tuple[..., str]`.

        - If keys are `partition`, they are assumed to be `partition` keys.
        - If keys are `*folds, partition`, then test whether for each fold there is a unique train partition.
        """
        if isinstance(self.folds, (Series, DataFrame)):
            split_index = self.folds.T.index
        elif isinstance(self.folds, Mapping):
            split_index = Index(self.folds.keys())
        else:
            raise TypeError(f"Cannot infer train-partition from {type(self.folds)=}")

        if isinstance(split_index, MultiIndex):
            *fold, partition = split_index.names
            df = split_index.to_frame(index=False)
            df["is_train"] = df[partition].str.lower().isin(self.train_patterns)
            if not all(df.groupby(fold)["is_train"].sum() == 1):
                raise ValueError("Each fold must have a unique train partition.")
        elif isinstance(split_index, Index):
            mask = split_index.str.lower().isin(self.train_patterns)
            if not sum(mask) == 1:
                raise ValueError("Each fold must have a unique train partition.")
        else:
            raise RuntimeError("Supposed to be unreachable")


class Inputs(NamedTuple):
    r"""Tuple of inputs."""

    t_target: Series
    x: DataFrame
    u: Optional[DataFrame] = None
    metadata: Optional[DataFrame] = None

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)


class Targets(NamedTuple):
    r"""Tuple of inputs."""

    y: DataFrame
    metadata: Optional[DataFrame] = None

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=False)


class Sample(NamedTuple):
    r"""A sample for forecasting task."""

    key: Hashable
    inputs: Inputs
    targets: Targets

    def __repr__(self) -> str:
        return repr_namedtuple(self, recursive=1)

    def sparsify_index(self) -> Sample:
        r"""Drop rows that contain only NAN values."""
        if self.inputs.x is not None:
            self.inputs.x.dropna(how="all", inplace=True)

        if self.inputs.u is not None:
            self.inputs.u.dropna(how="all", inplace=True)

        if self.targets.y is not None:
            self.targets.y.dropna(how="all", inplace=True)

        if self.inputs.t_target is not None:
            diff = self.inputs.t_target.index.difference(self.targets.y.index)
            self.inputs.t_target.drop(diff, inplace=True)

        return self


@dataclass
class TimeSeriesSampleGenerator(TorchDataset[Sample]):
    r"""Creates sample from a TimeSeriesCollection.

    There are different modus operandi for creating samples from a TimeSeriesCollection.

    Format Specification
    ~~~~~~~~~~~~~~~~~~~~

    - column-sparse
    - separate x and u and y

    - masked: In this format, two equimodal copies of the data are stored with appropriate masking.
        - inputs = (t, s, m)
        - targets = (s', m')

    - dense: Here, the data is split into groups of equal length. (x, u, y) share the same time index.
        - inputs = (t, x, u, m_x)
        - targets = (y, m_y)

    - sparse: Here, the data is split into groups sparse tensors. ALl NAN-only rows are dropped.
        - inputs = (t_y, (t_x, x), (t_u, u), m_x)
        - targets = (y, m_y)

    This class is used inside DataLoader.

    +---------------+------------------+------------------+
    | variable      | observation-mask | forecasting-mask |
    +===============+==================+==================+
    | observables X | ✔                | ✘                |
    +---------------+------------------+------------------+
    | controls U    | ✔                | ✔                |
    +---------------+------------------+------------------+
    | targets Y     | ✘                | ✔                |
    +---------------+------------------+------------------+

    Examples
    --------
    - time series classification task: empty forecasting horizon, only metadata_targets set.
    - time series imputation task: observation horizon and forecasting horizon overlap
    - time series forecasting task: observation horizon and forecasting horizon
    - time series forecasting task (autoregressive): observables = targets
    - time series event forecasting: predict both event and time of event (tᵢ, yᵢ)_{i=1:n} given n
    - time series event forecasting++: predict both event and time of event (tᵢ, yᵢ)_{i=1:n} and n

    Notes
    -----
    The option `pin_memory` of `torch.utils.data.DataLoader` recurses through
    Mappings and Sequence. However, it will cast the types. The only preserved Types are

    - dicts
    - tuples
    - namedtuples

    Dataclasses are currently not supported. Therefore, we preferably use namedtuples
    or dicts as containers.
    """

    dataset: TimeSeriesDataset | TimeSeriesCollection
    """The dataset to sample from."""

    _: KW_ONLY = NotImplemented
    targets: Index
    r"""Columns of the data that are used as targets."""
    observables: Index | list = NotImplemented
    r"""Columns of the data that are used as inputs."""
    covariates: Optional[Index | list] = None
    r"""Columns of the data that are used as controls."""
    metadata_targets: Optional[Index | list] = None
    r"""Columns of the metadata that are targets."""
    metadata_observables: Optional[Index | list] = NotImplemented
    r"""Columns of the metadata that are targets."""
    sample_format: Union[
        Literal["masked", "sparse"],
        tuple[Literal["masked", "sparse"], Literal["masked", "sparse"]],
    ] = "masked"

    def __post_init__(self) -> None:
        r"""Post init."""
        if self.observables is NotImplemented:
            self.observables = self.dataset.timeseries.columns
        if self.metadata_observables is NotImplemented:
            if self.dataset.metadata is None:
                self.metadata_observables = None
            else:
                self.metadata_observables = self.dataset.metadata.columns
        self.validate()

    def __iter__(self) -> Iterator[TS_Type_co]:
        return iter(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key: KeyVar) -> Sample:
        match self.sample_format:
            case "masked" | ("masked", "masked"):
                return self.make_sample(key, sparse_index=False, sparse_columns=False)
            case ("sparse", "masked"):
                return self.make_sample(key, sparse_index=True, sparse_columns=False)
            case ("masked", "sparse"):
                return self.make_sample(key, sparse_index=False, sparse_columns=True)
            case "sparse" | ("sparse", "sparse"):
                return self.make_sample(key, sparse_index=True, sparse_columns=True)
            case _:
                raise ValueError(f"Unknown sample format {self.sample_format=}")

    def __repr__(self) -> str:
        return repr_dataclass(self, recursive=1)

    def make_sample(
        self, key: KeyVar, *, sparse_index: bool = False, sparse_columns: bool = False
    ) -> Sample:
        r"""Create a sample from a TimeSeriesCollection."""
        # extract key
        if isinstance(self.dataset, TimeSeriesDataset):
            assert isinstance(key, tuple) and len(key) == 2
            observation_horizon, forecasting_horizon = key
            tsd = self.dataset
        elif isinstance(self.dataset, TimeSeriesCollection):
            assert isinstance(key, tuple) and len(key) == 2
            assert isinstance(key[1], Collection) and len(key[1]) == 2
            outer_key, (observation_horizon, forecasting_horizon) = key
            tsd = self.dataset[outer_key]
        else:
            raise NotImplementedError

        # timeseries
        ts_observed = tsd[observation_horizon]
        ts_forecast = tsd[forecasting_horizon]
        horizon_index = ts_observed.index.union(ts_forecast.index)
        ts = tsd[horizon_index]
        u: Optional[DataFrame] = None

        if sparse_columns:
            x = ts[self.observables].copy()
            x.loc[ts_forecast.index] = NA

            y = ts[self.targets].copy()
            y.loc[ts_observed.index] = NA

            if self.covariates is not None:
                u = ts[self.covariates].copy()
        else:
            x = ts.copy()
            # mask everything except covariates and observables
            columns = ts.columns.difference(self.covariates or [])
            x.loc[ts_observed.index, columns.difference(self.observables)] = NA
            x.loc[ts_forecast.index, columns] = NA

            y = ts.copy()
            # mask everything except targets in forecasting horizon
            y.loc[ts_observed.index] = NA
            y.loc[ts_forecast.index, ts.columns.difference(self.targets)] = NA

        # t_target
        t_target = y.index.to_series().copy()

        # metadata
        md = tsd.metadata
        md_targets: Optional[DataFrame] = None
        if self.metadata_targets is not None:
            assert md is not None
            md_targets = md[self.metadata_targets].copy()
            md = md.drop(columns=self.metadata_targets)

        # assemble sample
        inputs = Inputs(t_target=t_target, x=x, u=u, metadata=md)
        targets = Targets(y=y, metadata=md_targets)
        sample = Sample(key=key, inputs=inputs, targets=targets)

        if sparse_index:
            sample.sparsify_index()

        return sample

    def validate(self) -> None:
        r"""Validate that chosen columns are present."""
        ts = self.dataset.timeseries
        md = self.dataset.metadata
        observables = set(self.observables)
        targets = set(self.targets)
        covariates = set(self.covariates) if self.covariates is not None else set()
        md_observables = (
            set(self.metadata_observables)
            if self.metadata_observables is not None
            else set()
        )
        md_targets = (
            set(self.metadata_targets) if self.metadata_targets is not None else set()
        )
        ts_columns = set(ts.columns)
        md_columns = set(md.columns) if md is not None else set()

        if cols := covariates - ts_columns:
            raise ValueError(f"Covariates {cols} not in found timeseries columns!")
        if cols := observables - ts_columns:
            raise ValueError(f"Observables {cols} not in found timeseries columns!")
        if cols := targets - ts_columns:
            raise ValueError(f"Targets {cols} not found in timeseries columns!")
        if cols := covariates & observables:
            raise ValueError(f"Covariates and observables not disjoint! {cols}.")
        if cols := ts_columns - (observables | targets | covariates):
            warnings.warn(f"Unused columns in timeseries: {cols}")

        if md is not None:
            if cols := md_observables - md_columns:
                raise ValueError(f"Observables {cols} not in found metadata columns!")
            if cols := md_targets - md_columns:
                raise ValueError(f"Targets {cols} not in found metadata columns!")
