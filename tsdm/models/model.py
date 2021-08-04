r"""Base Model that all other models must subclass."""
import logging
import subprocess
from pathlib import Path
from urllib.parse import urlparse


logger = logging.getLogger(__name__)
__all__ = ["BaseModel"]


class BaseModel:
    r"""BaseModel that all models should subclass.

    Attributes
    ----------
    url: str
    name: str
    model_path: Path
    """

    url: str
    name: str
    model_path: Path

    def __init__(self):
        r"""TODO: add docstring."""
        raise NotImplementedError

    @classmethod
    def download(cls):
        r"""Download model (e.g. via git clone)."""
        model = cls.__name__
        parsed_url = urlparse(cls.url)
        logger.info("Obtaining model '%s' from %s", model, cls.url)

        if "google-research" in parsed_url.path:
            subprocess.run(
                f"svn export {cls.url} {cls.model_path}", shell=True, check=True
            )
            subprocess.run(
                f"grep -qxF '{cls.model_path}' .gitignore || echo '{cls.model_path}' >> .gitignore",
                shell=True,
                check=True,
            )
        else:
            subprocess.run(
                f"git clone {cls.url} {cls.model_path}", shell=True, check=True
            )
            # subprocess.run(F"git -C {model_path} pull", shell=True)

        logger.info("Finished importing model '%s' from %s", cls.name, cls.url)

    def predict(self, *inputs):
        r"""Synonym for forward and __call__."""
        raise NotImplementedError
