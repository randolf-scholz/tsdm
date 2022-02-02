r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "MLP",
]


import logging
from typing import Optional

from torch import nn

from tsdm.util.decorators import autojit

__logger__ = logging.getLogger(__name__)


@autojit
class MLP(nn.Sequential):
    """A standard Multi-Layer Perceptron."""

    HP: dict = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "output_size": int,
        "latent_size": int,
        "num_layers": 2,
        "dropout": float,
    }
    r"""Dictionary of hyperparameters."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        latent_size: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        self.dropout = dropout
        self.latent_size = input_size if latent_size is None else latent_size
        self.input_size = input_size
        self.output_size = output_size

        layers: list[nn.Module] = []

        # input layer
        layer = nn.Linear(self.input_size, self.latent_size)
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(layer.bias[None], nonlinearity="relu")
        layers.append(layer)

        # hidden layers
        for _ in range(num_layers - 1):
            layer = nn.Linear(self.latent_size, self.latent_size)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.kaiming_normal_(layer.bias[None], nonlinearity="relu")
            layers.append(layer)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        # output_layer
        layer = nn.Linear(self.latent_size, self.output_size)
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(layer.bias[None], nonlinearity="relu")
        layers.append(layer)
        super().__init__(*layers)
