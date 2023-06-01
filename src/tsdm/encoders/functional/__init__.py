r"""Implementation of encoders.

See Also:
    `tsdm.encoders` for modular implementations.
"""

__all__ = [
    # Types
    "FunctionalEncoder",
    # Constants
    # "SKLEARN_FUNCTIONAL_ENCODERS",
    "FUNCTIONAL_ENCODERS",
    # Functions
    "make_dense_triplets",
    "make_masked_format",
    "make_sparse_triplets",
    "time2float",
    "time2int",
    "triplet2dense",
    "timefeatures",
    # Functions from sklearn
    # "binarize",
    # "label_binarize",
    # "maxabs_scale",
    # "minmax_scale",
    # "normalize",
    # "power_transform",
    # "quantile_transform",
    # "robust_scale",
    # "scale",
]

from tsdm.encoders.functional._functional import (
    FunctionalEncoder,
    make_dense_triplets,
    make_masked_format,
    make_sparse_triplets,
    time2float,
    time2int,
    timefeatures,
    triplet2dense,
)

# from sklearn.preprocessing import (
#     binarize,
#     label_binarize,
#     maxabs_scale,
#     minmax_scale,
#     normalize,
#     power_transform,
#     quantile_transform,
#     robust_scale,
#     scale,
# )
# SKLEARN_FUNCTIONAL_ENCODERS: Final[dict[str, FunctionalEncoder]] = {
#     "binarize": binarize,
#     "label_binarize": label_binarize,
#     "maxabs_scale": maxabs_scale,
#     "minmax_scale": minmax_scale,
#     "normalize": normalize,
#     "power_transform": power_transform,
#     "quantile_transform": quantile_transform,
#     "robust_scale": robust_scale,
#     "scale": scale,
# }

FUNCTIONAL_ENCODERS: dict[str, FunctionalEncoder] = {
    "make_dense_triplets": make_dense_triplets,
    "make_masked_format": make_masked_format,
    "make_sparse_triplets": make_sparse_triplets,
    "time2float": time2float,
    "time2int": time2int,
    # "triplet2dense": triplet2dense,
}
r"""Dictionary of all available functional encoders."""
