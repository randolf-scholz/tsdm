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
    raise NotImplementedError


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
    'BRITS'         : load_brits,
    'mTAN'          : load_mtan,
    'M-RNN'         : load_mrnn,
    'NCDE'          : load_neural_cde,
    'Informer'      : load_informer,
    'TFT'           : load_tft,
}


def load_model(model: str):
    return model_loaders[model]()
