import importlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from .config import MODELDIR
from .model_cleaners import clean_model
from .model_io import download_model, model_available


@contextmanager
def add_to_path(p: Path) -> None:
    """Source: https://stackoverflow.com/a/41904558/9318372"""
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, str(p))
    try:
        yield
    finally:
        sys.path = old_path


def path_import(module_path: Path, module_name: str = None) -> ModuleType:
    """
    implementation taken from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    Source: https://stackoverflow.com/a/41904558/9318372

    Parameters
    ----------
    module_path: Path
        Path to the folder where the module is located
    module_name: str, optional

    Returns
    -------
    """

    module_name = module_name or module_path.parts[-1]
    module_init = module_path.joinpath("__init__.py")
    assert module_init.exists(), F"Module {module_path} has no __init__ file !!!"

    with add_to_path(module_path):
        spec = importlib.util.spec_from_file_location(module_name, str(module_init))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def load_neural_ode():
    raise NotImplementedError


def load_nbeats():
    raise NotImplementedError


def load_setts():
    raise NotImplementedError


def load_tpa():
    raise NotImplementedError


def load_gru_ode_bayes():
    raise NotImplementedError


def load_ip_net():
    raise NotImplementedError


def load_latent_ode():
    module = path_import(MODELDIR.joinpath('Latent-ODE'))
    return module.lib.latent_ode.LatentODE


def load_ode_rnn():
    module = path_import(MODELDIR.joinpath('ODE-RNN'))
    return module.lib.ode_rnn.ODE_RNN


def load_brits():
    raise NotImplementedError


def load_mtan():
    raise NotImplementedError


def load_mrnn():
    raise NotImplementedError


def load_neural_cde():
    raise NotImplementedError


def load_informer():
    raise NotImplementedError


def load_tft():
    raise NotImplementedError


model_loaders = {
    'NODE': load_neural_ode,
    'N-BEATS': load_nbeats,
    'SET-TS': load_setts,
    'TPA': load_tpa,
    'GRU-ODE-Bayes': load_gru_ode_bayes,
    'IP-Net': load_ip_net,
    'Latent-ODE': load_latent_ode,
    'ODE-RNN': load_ode_rnn,
    'BRITS': load_brits,
    'mTAN': load_mtan,
    'M-RNN': load_mrnn,
    'NCDE': load_neural_cde,
    'Informer': load_informer,
    'TFT': load_tft,
}


def load_model(model: str) -> type:
    """
    Load the specified model

    Parameters
    ----------
    model: str

    Returns
    -------
    class
    """
    assert model_available(model)

    if not MODELDIR.joinpath(model).exists():
        download_model(model)
        clean_model(model)

    return model_loaders[model]()
