r"""Base Model that all other models must subclass."""
import logging
import subprocess
from abc import abstractmethod
from pathlib import Path
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)
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
    model_path: Path

    def __init__(self):
        r"""TODO: add docstring."""
        raise NotImplementedError

    @classmethod
    def download(cls):
        r"""Download model (e.g. via git clone)."""
        parsed_url = urlparse(cls.url)
        LOGGER.info("Obtaining model '%s' from %s", cls.__name__, cls.url)

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

        LOGGER.info("Finished importing model '%s' from %s", cls.__name__, cls.url)

    @abstractmethod
    def __call__(self, *inputs):
        r"""Synonym for forward and __call__."""
        raise NotImplementedError
