import yaml
import logging
from pathlib import Path
import subprocess


with open("config/config.yaml") as fname:
    CONFIG = yaml.safe_load(fname)

with open("config/models.yaml") as fname:
    MODELS = yaml.safe_load(fname)

with open("config/datasets.yaml") as fname:
    DATASETS = yaml.safe_load(fname)

with open("config/hashes.yaml") as fname:
    HASHES = yaml.safe_load(fname)


HOMEDIR    = Path.home()
BASEDIR    = HOMEDIR.joinpath(CONFIG['basedir'])
LOGDIR     = BASEDIR.joinpath(CONFIG['logdir'])
MODELDIR   = BASEDIR.joinpath(CONFIG['modeldir'])
DATASETDIR = BASEDIR.joinpath(CONFIG['datasetdir'])
RAWDATADIR = BASEDIR.joinpath(CONFIG['rawdatadir'])


LOGDIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOGDIR.joinpath("example.log")),
    filemode="w",
    format="[%(asctime)s] [%(levelname)-s]\t[%(name)s]\t%(message)s",  # (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO)


def generate_folders(d: dict or str, current_path: Path) -> None:
    """
    Creates nested folder structure based on nested dictionary keys
    source: `StackOverflow <https://stackoverflow.com/a/22058144/9318372>`_

    Parameters
    ----------
    current_path: Path
    d: dict

    Returns
    -------
    None
    """
    for direc in d:
        path = current_path.joinpath(direc)
        if d[direc] is None:
            logging.info(F"creating folder {path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            generate_folders(d[direc], path)

    return


def available_models():
    return set().union(*[set(MODELS[source]) for source in MODELS])


def available_datasets():
    return set().union(*[set(DATASETS[source]) for source in DATASETS])


def download_model(model: str):
    if model.upper() == "ALL":
        for model in available_models():
            download_model(model)
        return

    logging.info(F"Importing {model=}")
    assert model in available_models(), F"Model {model} unknown. Available models: {available_models()}"
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


def cut_dirs(url: str):
    """automatically determine number of top directories to cut"""
    return url.count("/") - 3


def download_dataset(dataset: str, save_hash=True):
    if dataset.upper() == "ALL":
        for ds in available_datasets():
            download_dataset(ds)
        return

    logging.info(F"Importing {dataset=}")
    assert dataset in available_datasets(), F"Dataset {dataset} unknown. Available datasets: {available_datasets()}"
    dataset_path = RAWDATADIR.joinpath(dataset)
    dataset_path.mkdir(parents=True, exist_ok=True)

    if dataset in DATASETS['UCI']:
        url = DATASETS['UCI'][dataset]
        logging.info(F"Obtaining {dataset=} from UCI {url=}")
        subprocess.run(F"wget -r -np -nH -N --cut-dirs {cut_dirs(url)} -P '{dataset_path}' {url}", shell=True)
    elif dataset in DATASETS['MISC']:
        url = DATASETS['MISC'][dataset]
        logging.info(F"Obtaining {dataset=} from {url=}")
        subprocess.run(F"wget -r -np -nH -N --cut-dirs {cut_dirs(url)} -P '{dataset_path}' {url}", shell=True)
    elif dataset in DATASETS['GITHUB']:
        url = DATASETS['GITHUB'][dataset]
        logging.info(F"Obtaining {dataset=} from {url=}")
        subprocess.run(F"svn export {url.replace('tree/master', 'trunk')} {dataset_path}", shell=True)
    elif dataset in DATASETS['KAGGLE']:
        url = DATASETS['KAGGLE'][dataset]
        logging.info(F"Obtaining {dataset=} via Kaggle")
        subprocess.run(F"kaggle competitions download -p {dataset_path} -c {url}", shell=True)

    logging.info(F"Finished importing {dataset=}")
    if save_hash:
        with open(RAWDATADIR.joinpath("hashes.yaml"), "w+") as fname:
            hashes = yaml.safe_load(fname) or dict()
            hash = subprocess.run(
                F"sha1sum {dataset_path} | sha1sum | head -c 40",
                shell=True, encoding='utf-8', capture_output=True).stdout
            hashes[dataset] = hash
            yaml.safe_dump(hashes, fname)


logging.info("Initializing Folder Structure")
generate_folders(CONFIG['folders'], BASEDIR)
download_model("N-BEATS")

print(available_models())
print(available_datasets())

for dataset in DATASETS['KAGGLE']:
    download_dataset(dataset)


