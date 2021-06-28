"""
Base Model that all other models must subclass
"""
import logging
import subprocess
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class BaseModel:
    r"""BaseModel that all models should subclass"""
    url: str
    name: str
    model_path: Path

    def __init__(self):
        pass

    def predict(self, *inputs):
        """same as forward and __call__"""

    @classmethod
    def download(cls):
        """download model (e.g. via git clone)"""
        model = cls.__name__
        parsed_url = urlparse(cls.url)
        logger.info("Obtaining model '%s' from %s", model, cls.url)

        if "google-research" in parsed_url.path:
            subprocess.run(
                F"svn export {cls.url} {cls.model_path}",
                shell=True, check=True)
            subprocess.run(
                F"grep -qxF '{cls.model_path}' .gitignore || echo '{cls.model_path}' >> .gitignore",
                shell=True, check=True)
        else:
            subprocess.run(
                F"git clone {cls.url} {cls.model_path}",
                shell=True, check=True)
            # subprocess.run(F"git -C {model_path} pull", shell=True)

        logger.info("Finished importing model '%s' from %s", cls.name, cls.url)
