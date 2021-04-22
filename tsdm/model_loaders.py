import sys

from .config import MODELDIR
from .model_io import download_model
from .model_cleaners import clean_model
from importlib.machinery import SourceFileLoader


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
    model = MODELDIR.joinpath("Latent-ODE")
    if not model.exists():
        download_model('Latent-ODE')
        # clean_model('Latent-ODE')

    sys.path.insert(0, str(model))
    module = SourceFileLoader("models", str(model.joinpath("lib/latent_ode.py"))).load_module()
    LatentODE = getattr(module, 'LatentODE')
    return LatentODE


def load_ode_rnn():
    model = MODELDIR.joinpath("Latent-ODE")
    if not model.exists():
        download_model('Latent-ODE')
        clean_model('Latent-ODE')

    sys.path.insert(0, str(model))
    module = SourceFileLoader("models", str(model.joinpath("lib/ode_rnn.py"))).load_module()
    ODE_RNN = getattr(module, 'ODE_RNN')
    return ODE_RNN


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
    'NODE'          : load_neural_ode,
    'N-BEATS'       : load_nbeats,
    'SET-TS'        : load_setts,
    'TPA'           : load_tpa,
    'GRU-ODE-Bayes' : load_gru_ode_bayes,
    'IP-Net'        : load_ip_net,
    'Latent-ODE'    : load_latent_ode,
    'ODE-RNN'       : load_ode_rnn,
    'BRITS'         : load_brits,
    'mTAN'          : load_mtan,
    'M-RNN'         : load_mrnn,
    'NCDE'          : load_neural_cde,
    'Informer'      : load_informer,
    'TFT'           : load_tft,
}


def load_model(model: str):
    return model_loaders[model]()
