r"""Test scaled norm."""

import pytest
import torch

from tsdm import linalg
from tsdm.types.aliases import DimArg


@pytest.mark.parametrize("keepdim", [False, True], ids=lambda x: f"keepdim={x}")
@pytest.mark.parametrize("dims", [None, 0, (-1,), (0, 1)], ids=lambda x: f"axis={x}")
@pytest.mark.parametrize("shape", [(1, 2, 3)], ids=lambda x: f"shape={x}")
def test_shape(*, shape: tuple[int, ...], dims: DimArg, keepdim: bool) -> None:
    r"""Check that the output shape is correct."""
    torch.manual_seed(0)
    x = torch.randn(*shape)

    # convert axis to tuple of non-negative integers
    if dims is None:
        dim = tuple(range(x.ndim))
    elif isinstance(dims, int):
        dim = (dims,)
    else:
        dim = tuple(dims)
    dim = tuple(k % x.ndim for k in dim)

    if keepdim:
        reference_shape = tuple(1 if k in dim else shape[k] for k in range(x.ndim))
    else:
        reference_shape = tuple(shape[k] for k in range(x.ndim) if k not in dim)

    # compute reference
    reference_norm = x.norm(p=2, dim=dims, keepdim=keepdim)
    num_reduced = torch.tensor([shape[k] for k in dim]).prod()
    reference_scaled_norm = reference_norm / torch.sqrt(num_reduced)
    assert reference_norm.shape == reference_shape

    # compute norms
    scaled_norm = linalg.scaled_norm(x, p=2, axis=dims, keepdim=keepdim)
    tensor_norm = linalg.tensor_norm(x, p=2, axis=dims, keepdim=keepdim)
    assert scaled_norm.shape == reference_shape
    assert tensor_norm.shape == reference_shape
    assert torch.allclose(tensor_norm, reference_norm, atol=1e-6, rtol=1e-6)
    assert torch.allclose(scaled_norm, reference_scaled_norm, atol=1e-6, rtol=1e-6)
