from tsdm.random.samplers.base import BaseSampler, Sampler


def check_upcasting() -> None:
    r"""Check that BaseSampler can be upcast to Sampler."""

    def _upcast[T](arg: BaseSampler[T], /) -> Sampler[T]:
        return arg
