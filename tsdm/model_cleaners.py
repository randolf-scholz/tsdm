from .config import MODELDIR
from .model_io import download_model, model_available

__init_file = r'''"""
Automatically load all files in this folder
Source: https://stackoverflow.com/a/59054776/9318372
"""

from importlib import import_module
from pathlib import Path

__all__ = [
    import_module(f".{file.stem}", __package__)
    for file in Path(__file__).parent.glob("*.py")
    if "__" not in file.stem
]

del import_module, Path
'''


def clean_neural_ode():
    raise NotImplementedError


def clean_nbeats():
    raise NotImplementedError


def clean_setts():
    raise NotImplementedError


def clean_tpa():
    raise NotImplementedError


def clean_gru_ode_bayes():
    raise NotImplementedError


def clean_ip_net():
    raise NotImplementedError


def clean_latent_ode():
    model_path = MODELDIR.joinpath("Latent-ODE")
    model_init = model_path.joinpath("__init__.py")
    lib_path = model_path.joinpath("lib")
    lib_init = lib_path.joinpath("__init__.py")

    if not model_init.exists():
        with open(model_init, "w") as file:
            file.write("import lib\n")

    if not lib_init.exists():
        with open(lib_init, "w") as file:
            file.write(__init_file)


def clean_ode_rnn():
    model_path = MODELDIR.joinpath("ODE-RNN")
    model_init = model_path.joinpath("__init__.py")
    lib_path = model_path.joinpath("lib")
    lib_init = lib_path.joinpath("__init__.py")

    if not model_init.exists():
        with open(model_init, "w") as file:
            file.write("import lib\n")

    if not lib_init.exists():
        with open(lib_init, "w") as file:
            file.write(__init_file)


def clean_brits():
    raise NotImplementedError


def clean_mtan():
    raise NotImplementedError


def clean_mrnn():
    raise NotImplementedError


def clean_neural_cde():
    raise NotImplementedError


def clean_informer():
    raise NotImplementedError


def clean_tft():
    raise NotImplementedError


__model_cleaners = {
    'NODE'          : clean_neural_ode,
    'N-BEATS'       : clean_nbeats,
    'SET-TS'        : clean_setts,
    'TPA'           : clean_tpa,
    'GRU-ODE-Bayes' : clean_gru_ode_bayes,
    'IP-Net'        : clean_ip_net,
    'Latent-ODE'    : clean_latent_ode,
    'ODE-RNN'       : clean_ode_rnn,
    'BRITS'         : clean_brits,
    'mTAN'          : clean_mtan,
    'M-RNN'         : clean_mrnn,
    'NCDE'          : clean_neural_cde,
    'Informer'      : clean_informer,
    'TFT'           : clean_tft,
}


def clean_model(model: str):
    assert model_available(model)

    if not MODELDIR.joinpath(model).exists():
        download_model(model)

    return __model_cleaners[model]()
