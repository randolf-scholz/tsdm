r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "MLP",
]


import logging

from torch import nn

from tsdm.util.decorators import autojit

__logger__ = logging.getLogger(__name__)


@autojit
class MLP(nn.Sequential):
    """A standard Multi-Layer Perceptron."""

    def __init__(self, input_size: int, output_size: int, num_layers: int = 2):
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layer = nn.Linear(input_size, input_size)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.kaiming_normal_(layer.bias[None], nonlinearity="relu")
            layers.append(layer)
            layers.append(nn.ReLU())
        layer = nn.Linear(input_size, output_size)
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(layer.bias[None], nonlinearity="relu")
        layers.append(layer)
        super().__init__(*layers)
