r"""Custom processed version of the MIMIC-IV dataset."""

__all__ = ["MIMIC_IV_Scholz2024"]

from typing_extensions import Any

from tsdm.datasets.mimic.mimic_iv import MIMIC_IV


class MIMIC_IV_Scholz2024(MIMIC_IV):
    r"""Custom processed version of the MIMIC-IV dataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        r"""Initialize the dataset."""
        super().__init__(*args, **kwargs)
