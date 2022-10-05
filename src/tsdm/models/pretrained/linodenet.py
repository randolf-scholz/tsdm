r"""LinODEnet pretrained Models."""

__all__ = [
    # Classes
    "LinODEnet",
    # "LinODEnetEncoder",
]

from zipfile import ZipFile

import torch

from tsdm.models.pretrained.base import PreTrainedModel


class LinODEnet(PreTrainedModel):
    r"""Import pre-trained LinODEnet model."""

    model_file = "linodenet.zip"
    DOWNLOAD_URL = (
        "https://tubcloud.tu-berlin.de/s/syEZCZrBqQXiA5i/download/linodenet.zip"
    )
    MODEL_SHA256 = "15897965202b8e66db0189f4778655a3c55d350ca406447d8571133cbdfb1732"

    def _load(self) -> torch.nn.Module:
        with ZipFile(self.model_path) as archive:
            with archive.open("LinODEnet-70") as file:
                return self.load_torch_jit(file, map_location=self.device)


# class LinODEnetEncoder:
#     def __init__(self):
