def preprocess_neural_ode():
    raise NotImplementedError


def preprocess_nbeats():
    raise NotImplementedError


def preprocess_setts():
    raise NotImplementedError


def preprocess_tpa():
    raise NotImplementedError


def preprocess_gru_ode_bayes():
    raise NotImplementedError


def preprocess_ip_net():
    raise NotImplementedError


def preprocess_latent_ode():
    raise NotImplementedError


def preprocess_brits():
    raise NotImplementedError


def preprocess_mtan():
    raise NotImplementedError


def preprocess_mrnn():
    raise NotImplementedError


def preprocess_neural_cde():
    raise NotImplementedError


def preprocess_informer():
    raise NotImplementedError


def preprocess_tft():
    raise NotImplementedError


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
