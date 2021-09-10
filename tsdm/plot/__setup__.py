r"""Module Docstring."""
import logging
from typing import Final

import matplotlib

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = []

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{{amsmath}}"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["svg.fonttype"] = "none"
