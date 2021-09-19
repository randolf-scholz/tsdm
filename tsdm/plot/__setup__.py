r"""Module Docstring."""
import logging
from typing import Final

import matplotlib

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = []

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "pgf.texsystem": r"lualatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage{fontspec}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage{amsmath}",
                r"\usepackage{amsfonts}",
                r"\usepackage{amssymb}",
                r"\usepackage{unicode-math}",
            ]
        ),
        "text.latex.preamble": "\n".join(
            [
                r"\usepackage{amsmath}",
                r"\usepackage{amsfonts}",
                r"\usepackage{amssymb}",
            ]
        ),
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "svg.fonttype": "none",
    }
)
