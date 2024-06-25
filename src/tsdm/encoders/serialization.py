r"""Future serialization module.

Long Term Goals
---------------
- allow for serialization of encoders without pickling
    -> save parameters as files (JSON, npz, etc.)
    -> create a "from_parameters" classmethod that creates an encoder from parameters
    -> create a "serialize" method that saves the parameters to a file
    -> create a "deserialize" classmethod that loads the parameters from a file
"""

__all__ = ["serialize", "deserialize"]

import pickle
from pathlib import Path


def serialize(parameters: dict, *, path: Path) -> None:
    r"""Serialize parameters as zipfile.

    The

    - we collect all non-tensorial parameters in a file (json/yaml/toml),
      mapped as a dictionary.
    -

    Args:
        parameters: Parameters to serialize.
            We allow nested structures of the following types:
            - containers (dict/list/tuple)
              - dict keys must be valid python identifiers
            - tensors: numpy.ndarray/torch.Tensor/pandas.DataFrame/pandas.Series
                - tensors are serialized using functions provided by the respective
                  tensor libraries
            - leaf values: None/bool/int/float/str/etc...
              - everything that is not a container or tensor is considered a leaf
              - leaf values are serialized as json/yaml/toml

        path: Path to zipfile.
    """
    with path.open("wb") as f:
        pickle.dump(parameters, f)


def deserialize(path: Path) -> dict:
    r"""Deserialize parameters from zipfile.

    The file is traversed recursively, and the following rules are applied:
    - if the file is zip, apply deserialization recursively
    - if the file is json/yaml/toml, deserialize as dictionary
    - if the file is a tensor, deserialize using the respective tensor library
      - use the filename as key to the dictionary
    """
    with path.open("rb") as f:
        return pickle.load(f)
