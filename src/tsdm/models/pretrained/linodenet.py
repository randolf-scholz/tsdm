r"""LinODEnet pretrained Models."""

__all__ = [
    # Classes
    "LinODEnet",
]

from tsdm.models.pretrained.base import PreTrainedModel


class LinODEnet(PreTrainedModel):
    """Import pre-trained LinODEnet model."""

    model_file = ""
    download_url = ""
    MODEL_HASH = ""
