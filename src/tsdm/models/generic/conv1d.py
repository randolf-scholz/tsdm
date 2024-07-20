r"""Convolutional blocks for usage in a residual network."""

__all__ = ["ConvBlock"]

from collections import OrderedDict
from typing import Any, Literal

from torch import nn

from tsdm.utils.decorators import autojit

type PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]


@autojit
class ConvBlock(nn.Sequential):
    r"""A conv-block for usage in a residual network."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "num_subblocks": 2,
        "subblocks": [
            {
                "__name__": "ReLU",
                "__module__": "torch.nn",
                "inplace": False,
            },
            {
                "__name__": "Conv1d",
                "__module__": "torch.nn",
                "in_channels": None,
                "out_channels": None,
                "kernel_size": 16,
                "stride": 1,
                "padding": "same",
                "padding_mode": "replicate",
                "dilation": 1,
                "groups": 1,
                "bias": True,
            },
        ],
    }

    def __init__(
        self,
        input_size: int,
        *,
        kernel_size: int = 16,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: PaddingMode = "replicate",
        num_subblocks: int = 3,
        activation: Literal["ReLU", "leaky_relu", "Tanh", "Sigmoid"] = "ReLU",
    ) -> None:
        super().__init__()

        self.conv_kwargs: dict[str, Any] = {
            "in_channels": input_size,
            "out_channels": input_size,
            "kernel_size": kernel_size,
            "stride": 1,
            "padding": "same",
            "padding_mode": padding_mode,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
        }

        subblocks: OrderedDict[str, nn.Module] = OrderedDict()

        if not hasattr(nn, activation):
            raise ValueError(f"{activation=} not understood!")

        activation_class: type[nn.Module] = getattr(nn, activation)

        for k in range(num_subblocks):
            key = f"subblock{k}"
            module = nn.Sequential(
                activation_class(),
                nn.Conv1d(**self.conv_kwargs),
            )
            self.add_module(key, module)
            subblocks[key] = module

        super().__init__(subblocks)
