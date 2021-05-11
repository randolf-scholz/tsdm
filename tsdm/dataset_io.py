import logging
import subprocess
import warnings

import yaml

from .config import AVAILABLE_DATASETS, DATASETS, RAWDATADIR

logger = logging.getLogger(__name__)


def dataset_available(dataset: str) -> bool:
    r"""
    Checks whether the dataset is available.

    Parameters
    ----------
    dataset: str

    Returns
    -------
    bool
    """
    if dataset not in AVAILABLE_DATASETS:
        warnings.warn(F"{dataset=} unknown. {AVAILABLE_DATASETS=}")
        return False
    return True


def cut_dirs(url: str):
    r"""automatically determine number of top directories to cut"""
    return url.count("/") - 3


def download_dataset(dataset: str, save_hash=True) -> None:
    r"""
    Obtain dataset from the internet

    Parameters
    ----------
    dataset: str
    save_hash: bool, default=True
    """
    if dataset.upper() == "ALL":
        for ds in AVAILABLE_DATASETS:
            download_dataset(ds)
        return

    assert dataset_available(dataset)

    logger.info(F"Importing {dataset=}")
    dataset_path = RAWDATADIR.joinpath(dataset)
    dataset_path.mkdir(parents=True, exist_ok=True)

    if dataset in DATASETS['UCI']:
        url = DATASETS['UCI'][dataset]
        logger.info(F"Obtaining {dataset=} from UCI {url=}")
        subprocess.run(F"wget -r -np -nH -N --cut-dirs {cut_dirs(url)} -P '{dataset_path}' {url}", shell=True)
    elif dataset in DATASETS['MISC']:
        url = DATASETS['MISC'][dataset]
        logger.info(F"Obtaining {dataset=} from {url=}")
        subprocess.run(F"wget -r -np -nH -N --cut-dirs {cut_dirs(url)} -P '{dataset_path}' {url}", shell=True)
    elif dataset in DATASETS['GITHUB']:
        url = DATASETS['GITHUB'][dataset]
        logger.info(F"Obtaining {dataset=} from {url=}")
        subprocess.run(F"svn export {url.replace('tree/master', 'trunk')} {dataset_path}", shell=True)
    elif dataset in DATASETS['KAGGLE']:
        url = DATASETS['KAGGLE'][dataset]
        logger.info(F"Obtaining {dataset=} via Kaggle")
        subprocess.run(F"kaggle competitions download -p {dataset_path} -c {url}", shell=True)

    logger.info(F"Finished importing {dataset=}")
    if save_hash:
        with open(RAWDATADIR.joinpath("hashes.yaml"), "w+") as filename:
            hashes = yaml.safe_load(filename) or dict()
            hash_value = subprocess.run(
                F"sha1sum {dataset_path} | sha1sum | head -c 40",
                shell=True, encoding='utf-8', capture_output=True).stdout
            hashes[dataset] = hash_value
            yaml.safe_dump(hashes, filename)


def delete_dataset(dataset: str):
    r"""
    deletes downloaded dataset

    Parameters
    ----------
    dataset

    Returns
    -------

    """
    raise NotImplementedError


def validate_dataset(dataset: str):
    r"""
    Check dataset file integrity via hashing

    Parameters
    ----------
    dataset

    Returns
    -------

    """
    raise NotImplementedError
