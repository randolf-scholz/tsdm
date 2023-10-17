"""Test torch.jit compatibliity.."""

from tempfile import TemporaryFile
from typing import NamedTuple, Union

from torch import Tensor, jit, nn


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: Union[int, str]
    t: Tensor
    inputs: Tensor
    t_targets: Tensor
    targets: Tensor


def test_jit_namedtuple():
    """Test torch.jit compatibliity with namedtuple."""

    class Foo(nn.Module):
        def forward(self, x: Sample) -> Sample:
            return x

    model = Foo()

    scripted = jit.script(model)

    with TemporaryFile() as f:
        jit.save(scripted, f)
        f.seek(0)
        loaded = jit.load(f)

    loaded(Sample(1, Tensor(1), Tensor(1), Tensor(1), Tensor(1)))
    loaded(Sample("abs", Tensor(1), Tensor(1), Tensor(1), Tensor(1)))
