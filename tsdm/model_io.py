import logging
import subprocess
import warnings

import yaml

from .config import AVAILABLE_MODELS, MODELS, MODELDIR

logger = logging.getLogger(__name__)


def model_available(model: str) -> bool:
    r"""
    Checks whether the model is available.

    Parameters
    ----------
    model: str

    Returns
    -------
    bool
    """
    if model not in AVAILABLE_MODELS:
        warnings.warn(F"{model=} unknown. {AVAILABLE_MODELS=}")
        return False
    return True


def download_model(model: str, save_hash=True) -> None:
    r"""
    Obtain model from the internet.

    Parameters
    ----------
    model: str
    save_hash: bool, default=True
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

    logger.info(F"Finished importing {model=}")

    if save_hash:
        with open(MODELDIR.joinpath("hashes.yaml"), "w+") as filename:
            hashes = yaml.safe_load(filename) or dict()
            hash_value = subprocess.run(
                F"sha1sum {model_path} | sha1sum | head -c 40",
                shell=True, encoding='utf-8', capture_output=True).stdout
            hashes[model] = hash_value
            yaml.safe_dump(hashes, filename)
