r"""ODR-RNN Model Import."""

__all__ = [
    # Classes
    "ODE_RNN",
]

from pathlib import Path

import torch
from torch import Tensor, nn

from tsdm.models.base import BaseModel
from tsdm.utils import deep_dict_update
from tsdm.utils.system import import_module


class ODE_RNN(BaseModel, nn.Module):
    r"""ODE-RNN Model.

    Args:
        batch_size: Batch size
        classif_per_tp:
        concat_mask:
        device:
        input_dim: dimensionality of input
        lr: Learn-rate
        nonlinear: Nonlinearity used
        n_gru_units: Number of units per layer in each of GRU update networks
        n_labels: Number of outputs
        n_layers: iNumber of layers in ODE func in recognition ODE
        n_ode_gru_dims: Size of the latent state
        n_units: Number of units per layer in ODE func
        obsrv_std: Measurement error
        odeint_rtol: Relative tolerance of ODE solver
        odeint_atol: Absolute tolerance of ODE solver
        use_binary_classif:
        train_classif_w_reconstr:
        Net_cfg: Configuration parameters for the Net
        ODEFunc_cfg:Configuration parameters for the ODEFunc
        DiffeqSolver_cfg: Configuration parameters for the DiffeqSolver
        ODE_RNN_cfg: Configuration parameters for the ODE-RNN

    References:
        - https://papers.nips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html
    """

    url: str = r"https://github.com/YuliaRubanova/latent_ode.git"

    HP: dict = {
        # Size of the latent state
        "n_ode_gru_dims": 6,
        # Number of layers in ODE func in recognition ODE
        "n_layers": 1,
        # Number of units per layer in ODE func
        "n_units": 100,
        # nonlinearity used
        "nonlinear": nn.Tanh,
        # whether to concatenate mask to input
        "concat_mask": True,
        # dimensionality of input
        "input_dim": None,
        # device: 'cpu' or 'cuda'
        "device": torch.device("cpu"),
        # Number of units per layer in each of GRU update networks
        "n_gru_units": 100,
        # measurement error
        "obsrv_std": 0.01,
        # whether to use binary classification
        "use_binary_classif": False,
        # whether to train classifier with reconstruction loss
        "train_classif_w_reconstr": False,
        # whether to use classification per time point
        "classif_per_tp": False,
        # number of outputs
        "n_labels": 1,
        # relative tolerance of ODE solver
        "odeint_rtol": 1e-3,
        # absolute tolerance of ODE solver
        "odeint_atol": 1e-4,
        # batch_size
        "batch-size": 50,
        # learn-rate
        "lr": 1e-2,
        "ODEFunc_cfg": {},
        "DiffeqSolver_cfg": {},
        "ODE_RNN_cfg": {
            "input_dim": None,
            "latent_dim": None,
            "device": None,
        },
    }

    def __init__(
        self,
        *,
        input_dim: int,
        n_layers: int,
        n_ode_gru_dims: int,
        # optional args
        batch_size: int = 50,
        classif_per_tp: bool = False,
        concat_mask: bool = True,
        device: str | torch.device = "cpu",
        lr: float = 1e-2,
        method: str = "euler",
        n_gru_units: int = 100,
        n_labels: int = 1,
        n_units: int = 100,
        nonlinear: type[nn.Module] = nn.Tanh,
        obsrv_std: float = 0.01,
        odeint_atol: float = 1e-4,
        odeint_rtol: float = 1e-3,
        train_classif_w_reconstr: bool = False,
        use_binary_classif: bool = False,
        # cfg args
        DiffeqSolver_cfg: dict = NotImplemented,
        Net_cfg: dict = NotImplemented,
        ODEFunc_cfg: dict = NotImplemented,
        ODE_RNN_cfg: dict = NotImplemented,
    ) -> None:
        r"""Initialize the internal ODE-RNN model."""
        super().__init__()
        # TODO: Use tsdm.home_path or something
        module = import_module(Path.home() / ".tsdm/models/ODE-RNN")
        create_net = module.lib.utils.create_net
        ODEFunc = module.lib.ode_func.ODEFunc
        DiffeqSolver = module.lib.diffeq_solver.DiffeqSolver
        _ODE_RNN = module.lib.ode_rnn.ODE_RNN
        Net_cfg = {} if Net_cfg is NotImplemented else Net_cfg
        ODEFunc_cfg = {} if ODEFunc_cfg is NotImplemented else ODEFunc_cfg
        DiffeqSolver_cfg = (
            {} if DiffeqSolver_cfg is NotImplemented else DiffeqSolver_cfg
        )
        ODE_RNN_cfg = {} if ODE_RNN_cfg is NotImplemented else ODE_RNN_cfg

        HP = {
            "input_dim": input_dim,
            "n_layers": n_layers,
            "n_ode_gru_dims": n_ode_gru_dims,
            # optional args
            "batch_size": batch_size,
            "classif_per_tp": classif_per_tp,
            "concat_mask": concat_mask,
            "device": device,
            "lr": lr,
            "method": method,
            "n_gru_units": n_gru_units,
            "n_labels": n_labels,
            "n_units": n_units,
            "nonlinear": nonlinear,
            "obsrv_std": obsrv_std,
            "odeint_atol": odeint_atol,
            "odeint_rtol": odeint_rtol,
            "train_classif_w_reconstr": train_classif_w_reconstr,
            "use_binary_classif": use_binary_classif,
            # cfg args
            "DiffeqSolver_cfg": DiffeqSolver_cfg,
            "Net_cfg": Net_cfg,
            "ODEFunc_cfg": ODEFunc_cfg,
            "ODE_RNN_cfg": ODE_RNN_cfg,
        }
        self.HP = deep_dict_update(self.HP, HP)

        self.ode_func_net = create_net(
            n_inputs=n_ode_gru_dims,
            n_outputs=n_ode_gru_dims,
            n_layers=n_layers,
            n_units=n_units,
            nonlinear=nonlinear,
        )

        self.rec_ode_func = ODEFunc(
            ode_func_net=self.ode_func_net,
            input_dim=input_dim,
            latent_dim=n_ode_gru_dims,
            device=device,
        )

        self.z0_diffeq_solver = DiffeqSolver(
            input_dim=input_dim,
            ode_func=self.rec_ode_func,
            method=method,
            latents=n_ode_gru_dims,
            odeint_rtol=odeint_rtol,
            odeint_atol=odeint_atol,
            device=device,
        )

        self.model = _ODE_RNN(
            input_dim=input_dim,
            latent_dim=n_ode_gru_dims,
            device=device,
            z0_diffeq_solver=self.z0_diffeq_solver,
            n_gru_units=n_gru_units,
            concat_mask=concat_mask,
            obsrv_std=obsrv_std,
            use_binary_classif=use_binary_classif,
            classif_per_tp=classif_per_tp,
            n_labels=n_labels,
            train_classif_w_reconstr=train_classif_w_reconstr,
        )

    def forward(self, T: Tensor, X: Tensor) -> Tensor:
        r"""TODO: add docstring."""
        (pred,) = self.model.get_reconstruction(
            # Note: n_traj_samples and mode have no effect -> omitted!
            time_steps_to_predict=T,
            data=X,
            truth_time_steps=T,
            mask=torch.isnan(X),
        )

        return pred
