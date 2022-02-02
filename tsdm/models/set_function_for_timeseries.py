r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # CLASSES
    "SetFuncTS",
]

import logging
from typing import Optional

import torch
from torch import Tensor, jit, nn

from tsdm.encoders.modular.torch import PositionalEncoder
from tsdm.models.generic import MLP, DeepSet, ScaledDotProductAttention
from tsdm.util.decorators import autojit

__logger__ = logging.getLogger(__name__)


@autojit
class SetFuncTS(nn.Module):
    r"""Set function for time series.

    References
    ----------
    - | Set Functions for Time Series
      | Max Horn, Michael Moor, Christian Bock, Bastian Rieck, Karsten Borgwardt
      | Proceedings of the 37th International Conference on Machine Learning
      | PMLR 119:4353-4363, 2020.
      | https://proceedings.mlr.press/v119/horn20a.html
    - https://github.com/BorgwardtLab/Set_Functions_for_Time_Series
    """

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "time_encoder": PositionalEncoder.HP,
        "key_encoder": DeepSet.HP,
        "value_encoder": MLP.HP,
        "attention": ScaledDotProductAttention.HP,
        "head": MLP.HP,
    }
    r"""Dictionary of hyperparameters."""

    # BUFFER
    dummy: Tensor

    def __init__(
        self,
        input_size: int,
        output_size: int,
        latent_size: Optional[int] = None,
        dim_keys: Optional[int] = None,
        dim_vals: Optional[int] = None,
        dim_time: Optional[int] = None,
    ) -> None:
        super().__init__()

        dim_keys = input_size if dim_keys is None else dim_keys
        dim_vals = input_size if dim_vals is None else dim_vals
        dim_time = 8 if dim_time is None else dim_time
        latent_size = input_size if latent_size is None else latent_size
        # time_encoder
        # feature_encoder -> CNN?
        self.time_encoder = PositionalEncoder(dim_time, scale=10.0)
        self.key_encoder = DeepSet(input_size + dim_time - 1, dim_keys)
        print(dim_vals)
        self.value_encoder = MLP(input_size + dim_time - 1, dim_vals)
        self.attn = ScaledDotProductAttention(
            dim_keys + input_size + dim_time - 1, dim_vals, latent_size
        )
        self.head = MLP(latent_size, output_size)
        self.register_buffer("dummy", torch.zeros(1))

    @jit.export
    def forward(self, t: Tensor, v: Tensor, m: Tensor) -> Tensor:
        r"""Signature: `(..., T, D) → (..., F)`.

        s must be a tensor of the shape L×(2+C), sᵢ = [tᵢ, zᵢ, mᵢ], where
        - tᵢ is timestamp
        - zᵢ is observed value
        - mᵢ is identifier

        C is the number of classes (one-hot encoded identifier)

        Parameters
        ----------
        t: Tensor `[*V, T]`
        v: Tensor `[*V, D]`
        m: Tensor `[*V, K]`

        Returns
        -------
        Tensors [F]
        """
        t = t.to(device=self.dummy.device)
        v = v.to(device=self.dummy.device)
        m = m.to(device=self.dummy.device)

        time_features = self.time_encoder(t)

        if v.ndim < m.ndim:
            v = v.unsqueeze(-1)

        s = torch.cat([time_features, v, m], dim=-1)
        fs = self.key_encoder(s)
        fs = torch.tile(fs.unsqueeze(-2), (s.shape[-2], 1))
        K = torch.cat([fs, s], dim=-1)
        V = self.value_encoder(s)
        mask = torch.isnan(s[..., 0])
        z = self.attn(K, V, mask=mask)
        y = self.head(z)
        return y

    @jit.export
    def forward_tuple(self, t: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        r"""Signature: `(∗V, T), (*V, D), (*V, K) → (..., F)`."""
        return self.forward(t[0], t[1], t[2])

    @jit.export
    def forward_batch(self, batch: list[tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        r"""Signature: `[(∗V, T), (*V, D), (*V, K)] → (..., F)`.

        Parameters
        ----------
        batch: list[tuple[Tensor, Tensor, Tensor]

        Returns
        -------
        Tensor
        """
        return torch.cat([self.forward(t, v, m) for t, v, m in batch])
