r"""ReZero module."""

__all__ = [
    "ReZero",
    "ReZeroMLP",
    "ResNetBlock",
    "ConcatEmbedding",
]

from collections import OrderedDict
from math import ceil, log2
from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn
from torch._jit_internal import _copy_to_script_wrapper

from tsdm.models.generic.dense import ReverseDense
from tsdm.utils import deep_dict_update, initialize_from_config
from tsdm.utils.decorators import autojit


@autojit
class ResNetBlock(nn.Sequential):
    r"""Pre-activation ResNet block.

    References:
        - | Identity Mappings in Deep Residual Networks
          | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
          | European Conference on Computer Vision 2016
          | https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38
    """

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "num_subblocks": 2,
        "subblocks": [
            ReverseDense.HP,
        ],
    }

    def __init__(self, **HP: Any) -> None:
        super().__init__()

        self.CFG = HP = deep_dict_update(self.HP, HP)

        if HP["input_size"] is None:
            raise ValueError("input_size is required!")

        for layer in HP["subblocks"]:
            if layer["__name__"] == "Linear":
                layer["in_features"] = HP["input_size"]
                layer["out_features"] = HP["input_size"]
            if layer["__name__"] == "BatchNorm1d":
                layer["num_features"] = HP["input_size"]
            else:
                layer["input_size"] = HP["input_size"]
                layer["output_size"] = HP["input_size"]

        subblocks: OrderedDict[str, nn.Module] = OrderedDict()

        for k in range(HP["num_subblocks"]):
            key = f"subblock{k}"
            module = nn.Sequential(*[
                initialize_from_config(layer) for layer in HP["subblocks"]
            ])
            self.add_module(key, module)
            subblocks[key] = module

        super().__init__(subblocks)


@autojit
class ReZero(nn.Sequential):
    r"""A ReZero model."""

    weights: Tensor
    r"""PARAM: The weights of the model."""

    def __init__(self, *blocks: nn.Module, weights: Optional[Tensor] = None) -> None:
        super().__init__(*blocks)
        w = torch.zeros(len(blocks)) if weights is None else weights
        self.register_parameter("weights", nn.Parameter(w.to(torch.float)))

    @jit.export
    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        r""".. signature:: ``(..., n) -> (..., n)``."""
        for k, block in enumerate(self):
            x = x + self.weights[k] * block(x)
        return x

    @_copy_to_script_wrapper
    def __getitem__(self, item: int | slice) -> nn.Module:
        r"""Get a submodel."""
        modules: list[nn.Module] = list(self._modules.values())
        if isinstance(item, slice):
            return ReZero(*modules[item], weights=self.weights[item])
        return modules[item]

    @jit.export
    def __len__(self) -> int:
        r"""Get the number of submodels."""
        return len(self._modules)


@autojit
class ConcatEmbedding(nn.Module):
    r"""Maps $x ⟼ [x,w]$."""

    HP = {
        "__name__": __qualname__,
        "__doc__": __doc__,
        "__module__": __name__,
        "input_size": int,
        "hidden_size": int,
    }
    r"""Dictionary of Hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    pad_size: Final[int]
    r"""CONST: The size of the padding."""

    # BUFFERS
    scale: Tensor
    r"""BUFFER: The scaling scalar."""

    # Parameters
    padding: Tensor
    r"""PARAM: The padding vector."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size > hidden_size:
            raise ValueError(
                f"ConcatEmbedding requires {input_size=} <= {hidden_size=}!"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pad_size = hidden_size - input_size
        self.padding = nn.Parameter(torch.randn(self.pad_size))

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. signature:: ``(..., d) -> (..., d+e)``."""
        shape = list(x.shape[:-1]) + [self.pad_size]  # noqa: RUF005
        z = torch.cat([x, self.padding.expand(shape)], dim=-1)
        torch.cuda.synchronize()  # needed when `cat` holds 0-size tensor
        return z

    @jit.export
    def inverse(self, z: Tensor) -> Tensor:
        r""".. signature:: ``(..., d+e) -> (..., d)``.

        The reverse of the forward. Satisfies inverse(forward(x)) = x for any input.
        """
        return z[..., : self.input_size]


@autojit
class ReZeroMLP(nn.Sequential):
    r"""A ReZero based on MLP and Encoder + Decoder."""

    latent_size: Final[int]
    r"""CONST: The dimensionality of the latent space."""
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        latent_size: Optional[int] = None,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = (
            2 ** ceil(log2(input_size)) if latent_size is None else latent_size
        )

        self.encoder = ConcatEmbedding(self.input_size, self.latent_size)

        blocks = [
            nn.Sequential(
                ReverseDense(self.latent_size, self.latent_size // 2),
                ReverseDense(self.latent_size // 2, self.latent_size),
            )
            for _ in range(num_blocks)
        ]

        self.blocks = ReZero(*blocks)
        self.decoder = ReverseDense(
            input_size=self.latent_size, output_size=self.output_size
        )

        super().__init__(*[self.encoder, self.blocks, self.decoder])
