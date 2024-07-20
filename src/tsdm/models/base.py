r"""Base Model that all other models must subclass."""

__all__ = [
    # ABCs & Protocols
    "Model",
    "ForecastingModel",
    "StateSpaceForecastingModel",
    # Classes
    "BaseModelMetaClass",
    "BaseModel",
]

import logging
import os
import subprocess
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Optional, Protocol
from urllib.parse import urlparse

from torch import Tensor, nn

from tsdm.config import CONFIG

type Model = nn.Module
r"""Type hint for models."""


class ForecastingModel(Protocol):
    r"""Generic forecasting model."""

    def predict(
        self,
        q: Tensor,
        X: tuple[Tensor, Tensor],
        U: Optional[tuple[Tensor, Tensor]] = None,
        M: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Return the actual forecast x(t) for time t.

        Args:
            q: Query time at which to make the forecast.
            X: Tuple of tensors (t, x) containing the time and observation of the system. (observables)
            U: Tuple of tensors (t, u) containing the time and control input of the system. (covariates)
            M: Static metadata for the system. (time-invariant covariates)

        Returns:
            x(t): Forecast for time t.
        """
        ...


class StateSpaceForecastingModel(ForecastingModel, Protocol):
    r"""State Space forecasting model."""

    def predict(
        self,
        q: Tensor,
        X: tuple[Tensor, Tensor],
        U: Optional[tuple[Tensor, Tensor]] = None,
        M: Optional[Tensor] = None,
        t0: Optional[Tensor] = None,
        z0: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Return the encoded forecast x(t) for time t.

        Args:
            q: Query time at which to make the forecast.
            X: Tuple of tensors (t, x) containing the time and observation of the system. (observables)
            U: Tuple of tensors (t, u) containing the time and control input of the system. (covariates)
            M: Static metadata for the system. (time-invariant covariates)
            t0: Initial time of the system.
            z0: Initial (latent) state of the system.

        Returns:
            y(q | (t, x), (t, u), m): Forecast for time t given the system state.
        """
        ...


class BaseModelMetaClass(type):
    r"""Metaclass for BaseDataset."""

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwds: Any,
    ) -> None:
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if os.environ.get("GENERATING_DOCS", False):
            cls.MODEL_DIR = Path(f"~/.tsdm/models/{cls.__name__}/")
        else:
            cls.MODEL_DIR = CONFIG.MODELDIR / cls.__name__


class BaseModel(metaclass=BaseModelMetaClass):
    r"""BaseModel that all models should subclass."""

    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the model."""
    SOURCE_URL: Optional[str] = None
    r"""HTTP address from where the model can be downloaded."""
    INFO_URL: Optional[str] = None
    r"""HTTP address containing additional information about the dataset."""
    MODEL_DIR: Path
    r"""Location where the model is stored."""

    @cached_property
    def model_path(self) -> Path:
        r"""Return the path to the model."""
        return CONFIG.MODELDIR / self.__class__.__name__

    def download(self, *, url: Optional[str | Path] = None) -> None:
        r"""Download model (e.g. via git clone)."""
        target_url: str = str(self.SOURCE_URL) if url is None else str(url)
        parsed_url = urlparse(target_url)

        self.LOGGER.info("Obtaining model from %s", self.SOURCE_URL)

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
                f"svn export {self.SOURCE_URL} {self.model_path}",
                shell=True,
                check=True,
            )
            subprocess.run(
                f"grep -qxF {self.model_path!r} .gitignore || echo"
                f" {self.model_path!r} >> .gitignore",
                shell=True,
                check=True,
            )
        else:
            subprocess.run(
                f"git clone {self.SOURCE_URL} {self.model_path}", shell=True, check=True
            )

        self.LOGGER.info("Finished importing model from %s", self.SOURCE_URL)
