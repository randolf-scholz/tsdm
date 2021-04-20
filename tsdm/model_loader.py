def preprocess_neural_ode():
    pass


def preprocess_nbeats():
    pass


def preprocess_setts():
    pass


def preprocess_tpa():
    pass


def preprocess_gru_ode_bayes():
    pass


def preprocess_ip_net():
    pass


def preprocess_latent_ode():
    pass


def preprocess_brits():
    pass


def preprocess_mtan():
    pass


def preprocess_mrnn():
    pass


def preprocess_neural_cde():
    pass


def preprocess_informer():
    pass


def preprocess_tft():
    pass


model_preprocessors = {
    'NODE'          : preprocess_neural_ode,
    'N-BEATS'       : preprocess_nbeats,
    'SET-TS'        : preprocess_setts,
    'TPA'           : preprocess_tpa,
    'GRU-ODE-Bayes' : preprocess_gru_ode_bayes,
    'IP-Net'        : preprocess_ip_net,
    'Latent-ODE'    : preprocess_latent_ode,
    'BRITS'         : preprocess_brits,
    'mTAN'          : preprocess_mtan,
    'M-RNN'         : preprocess_mrnn,
    'NCDE'          : preprocess_neural_cde,
    'Informer'      : preprocess_informer,
    'TFT'           : preprocess_tft,
}


def preprocess_model(model: str):
    return model_preprocessors[model]()
