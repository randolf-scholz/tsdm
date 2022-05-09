r"""Base Model that all other models must subclass."""

__all__ = [
    # Classes
    "BaseModel",
]

import logging
import subprocess
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

from tsdm.config import MODELDIR

__logger__ = logging.getLogger(__name__)


class BaseModel(ABC):
    r"""BaseModel that all models should subclass.

    Attributes
    ----------
    url: str
    name: str
    model_path: Path
    """

    url: str

    @cached_property
    def model_path(self) -> Path:
        r"""Return the path to the model."""
        return MODELDIR / self.__class__.__name__

    def download(self, *, url: Optional[Union[str, Path]] = None) -> None:
        r"""Download model (e.g. via git clone)."""
        target_url: str = str(self.url) if url is None else str(url)
        parsed_url = urlparse(target_url)

        __logger__.info(
            "Obtaining model '%s' from %s", self.__class__.__name__, self.url
        )

        if parsed_url.netloc == "github.com":

            if "tree/main" in target_url:
                export_url = target_url.replace("tree/main", "trunk")
            elif "tree/master" in target_url:
                export_url = target_url.replace("tree/master", "trunk")
            else:
                raise ValueError(f"Unrecognized URL: {target_url}")

            subprocess.run(
                f"svn export --force {export_url} {self.model_path}",
                shell=True,
                check=True,
            )
        elif "google-research" in parsed_url.path:
            subprocess.run(
                f"svn export {self.url} {self.model_path}", shell=True, check=True
            )
            subprocess.run(
                f"grep -qxF '{self.model_path}' .gitignore || echo '{self.model_path}' >> .gitignore",
                shell=True,
                check=True,
            )
        else:
            subprocess.run(
                f"git clone {self.url} {self.model_path}", shell=True, check=True
            )
            # subprocess.run(F"git -C {model_path} pull", shell=True)

        __logger__.info(
            "Finished importing model '%s' from %s", self.__class__.__name__, self.url
        )

    @abstractmethod
    def forward(self):
        r"""Synonym for forward and __call__."""
        raise NotImplementedError
