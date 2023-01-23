#!/usr/bin/env python
"""Test linodenet initialization methods."""

from pathlib import Path

from torch import jit


def generate_checkpoint() -> Path:
    r"""Generate a checkpoint."""
    from linodenet.models import LinODEnet, ResNet  # type: ignore[import]

    MODEL_CONFIG = {
        "__name__": "LinODEnet",
        "input_size": 2,
        "hidden_size": 3,
        "latent_size": 4,
        "Encoder": ResNet.HP | {"num_blocks": 2},
        "Decoder": ResNet.HP | {"num_blocks": 2},
    }

    PATH = Path("/tmp") / "test_linodenet.pt"
    model = LinODEnet(**MODEL_CONFIG)
    scripted = jit.script(model)
    jit.save(scripted, PATH)
    return PATH


# def test_from_zipfile() -> None:
#     r"""Test if pretrained linodenet mode can be loaded from checkpoint."""
#     PATH = generate_checkpoint()
#     model = LinODEnet.from_zipfile(PATH)
#     assert isinstance(model, LinODEnet)


# def test_from_url() -> None:
#     r"""Test if pretrained linodenet mode can be loaded from URL."""
#     model = LinODEnet.from_url()
#     assert isinstance(model, LinODEnet)


# def test_from_checkpoint() -> None:
#     r"""Test if pretrained linodenet mode can be loaded from URL."""
#     for checkpoint in LinODEnet.CHECKPOINTS:
#         model = LinODEnet.from_checkpoint(checkpoint)
#         assert isinstance(model, LinODEnet)


if __name__ == "__main__":
    pass
