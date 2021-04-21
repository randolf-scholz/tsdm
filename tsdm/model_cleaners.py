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
    raise NotImplementedError


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


model_cleanors = {
  'NODE'          : clean_neural_ode,
  'N-BEATS'       : clean_nbeats,
  'SET-TS'        : clean_setts,
  'TPA'           : clean_tpa,
  'GRU-ODE-Bayes' : clean_gru_ode_bayes,
  'IP-Net'        : clean_ip_net,
  'Latent-ODE'    : clean_latent_ode,
  'BRITS'         : clean_brits,
  'mTAN'          : clean_mtan,
  'M-RNN'         : clean_mrnn,
  'NCDE'          : clean_neural_cde,
  'Informer'      : clean_informer,
  'TFT'           : clean_tft,

}


def clean_model(model: str):
    return model_cleanors[model]()
