import pytest
import torch


@pytest.mark.parametrize(
    "reduction",
    [
        torch.sum,
        torch.mean,
        torch.nanmean,
        torch.nansum,
        torch.nanmedian,
        torch.prod,
        torch.amax,
        torch.amin,
        torch.std,
        torch.var,
    ],
)
@pytest.mark.xfail(reason="https://github.com/pytorch/pytorch/issues/29137")
def test_reduction(reduction) -> None:
    arg = torch.randn(2, 3, 4)

    expected = arg.shape
    result = reduction(arg, dim=())
    actual = result.shape

    assert actual == expected
