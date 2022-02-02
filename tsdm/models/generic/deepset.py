r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "DeepSet",
]

import logging
from typing import Optional

import torch
from torch import Tensor, nn

from tsdm.models.generic.mlp import MLP
from tsdm.util.decorators import autojit

__logger__ = logging.getLogger(__name__)


@autojit
class DeepSet(nn.Module):
    r"""Permutation invariant deep set model."""

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "output_size": int,
        "latent_size": int,
        "encoder": MLP.HP,
        "decoder": MLP.HP,
    }
    r"""Dictionary of hyperparameters."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        latent_size: Optional[int] = None,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        # aggregation: Literal["min", "max", "sum", "mean", "prod"] = "sum",
    ):
        super().__init__()
        latent_size = input_size if latent_size is None else latent_size
        self.encoder = MLP(input_size, latent_size, encoder_layers)
        self.decoder = MLP(latent_size, output_size, decoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        r"""Signature: `[..., <Var>, D] -> [..., F]`.

        Components:
          - Encoder: [..., D] -> [..., E]
          - Aggregation: [..., V, E] -> [..., E]
          - Decoder: [..., E] -> [..., F]

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        x = self.encoder(x)
        x = torch.nanmean(x, dim=-2)
        x = self.decoder(x)
        return x
