r"""LinODEnet pretrained Models."""

__all__ = [
    # Classes
    "OldLinODEnet",
    "LinODEnet",
]

from tsdm.models.pretrained.base import PreTrainedModel


class OldLinODEnet(PreTrainedModel):  # Deprecated
    """Import pre-trained LinODEnet model."""

    rawdata_file = "linodenet.zip"
    DOWNLOAD_URL = (
        "https://tubcloud.tu-berlin.de/s/syEZCZrBqQXiA5i/download/linodenet.zip"
    )
    RAWDATA_HASH = "15897965202b8e66db0189f4778655a3c55d350ca406447d8571133cbdfb1732"
    HASHES = {
        "model": ...,
        "encoder": ...,
        "optimizer": ...,
    }
    component_files = {
        "model": "LinODEnet-70",
        "encoder": "encoder.pickle",
    }


class LinODEnet(PreTrainedModel):  # Deprecated
    """Import pre-trained LinODEnet model."""

    rawdata_file = "2022-11-16-linodenet-e4f9e3bd1e93ff868a0c400dee58d5e9.zip"
    DOWNLOAD_URL = (
        "https://tubcloud.tu-berlin.de/s/njNwW3gkFtwAiXZ/download/"
        "2022-11-16-linodenet-e4f9e3bd1e93ff868a0c400dee58d5e9.zip"
    )
    RAWDATA_HASH = "2939a2528eb791d601e7433894e7ca775dec00a53483333d995d0c914a434e6d"
    component_files = {
        "model": "RecursiveScriptModule-30",
        "encoder": "encoder.pickle",
        "optimizer": "AdamW-30",
        "hyperparameters": "hparams.yaml",
    }
