r"""#TODO add module summary line.

#TODO add module description.
"""

__all__ = [
    # Classes
    "MLP",
]


from torch import nn
from typing_extensions import Optional

from tsdm.utils.wrappers import autojit


@autojit
class MLP(nn.Sequential):
    r"""A standard Multi-Layer Perceptron."""

    HP: dict = {
        "__name__": __qualname__,
        "__doc__": __doc__,
        "__module__": __name__,
        "inputs_size": int,
        "output_size": int,
        "hidden_size": int,
        "num_layers": 2,
        "dropout": float,
    }
    r"""Dictionary of hyperparameters."""

    def __init__(
        self,
        inputs_size: int,
        output_size: int,
        *,
        hidden_size: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        self.dropout = dropout
        self.hidden_size = inputs_size if hidden_size is None else hidden_size
        self.inputs_size = inputs_size
        self.output_size = output_size

        # build layers
        layers: list[nn.Module] = []

        # input layer (change shape to hidden_size)
        linear = nn.Linear(self.inputs_size, self.hidden_size)
        nn.init.kaiming_normal_(linear.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(linear.bias[None], nonlinearity="linear")
        layers.append(linear)

        # hidden layers (pre-activation + dropout + linear)
        for _ in range(num_layers - 1):
            # init layers
            linear = nn.Linear(self.hidden_size, self.hidden_size)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.kaiming_normal_(linear.bias[None], nonlinearity="relu")
            layers.extend([nn.ReLU(), nn.Dropout(self.dropout), linear])

        # output_layer (pre-activation + dropout + change shape to output_size)
        linear = nn.Linear(self.hidden_size, self.output_size)
        nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(linear.bias[None], nonlinearity="relu")
        layers.extend([nn.ReLU(), nn.Dropout(self.dropout), linear])

        super().__init__(*layers)
