import logging
import subprocess

from .config import AVAILABLE_MODELS, MODELS, MODELDIR


def model_available(model: str):
    if model not in AVAILABLE_MODELS:
        raise NotImplementedError(F"{model=} unknown. {AVAILABLE_MODELS=}")
    return True


def download_model(model: str):
    """
    Obtain Model from the internet

    Parameters
    ----------
    model: str

    Returns
    -------

    """
    if model.upper() == "ALL":
        for model in AVAILABLE_MODELS:
            download_model(model)
        return

    assert model_available(model)

    logging.info(F"Importing {model=}")
    model_path = MODELDIR.joinpath(model)
    model_path.mkdir(parents=True, exist_ok=True)

    if model in MODELS['github']:
        url = str(MODELS['github'][model])
        subprocess.run(F"git clone {url} {model_path}", shell=True)
        # subprocess.run(F"git -C {model_path} pull", shell=True)
    elif model in MODELS['google']:
        url = MODELS['google'][model]
        subprocess.run(F"svn export {url} {model_path}")
        subprocess.run(F"grep -qxF '{model_path}' .gitignore || echo '{model_path}' >> .gitignore")
