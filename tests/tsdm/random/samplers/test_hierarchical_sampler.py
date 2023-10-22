r"""Test `module.class`."""

from tsdm.random.samplers import HierarchicalSampler, RandomSampler


def test_hierarchical_sampler() -> None:
    data = {
        "foo": {"a": 1, "b": 2},
        "bar": {"x": 3, "y": 4, "z": 5},
    }

    key_pairs = {(outer, inner) for outer in data for inner in data[outer]}

    subsamplers = {
        key: RandomSampler(value, shuffle=True) for key, value in data.items()
    }

    sampler = HierarchicalSampler(data, subsamplers=subsamplers, shuffle=True)

    for key in sampler:
        print(key)

    assert len(sampler) == len(key_pairs)
    assert set(sampler) == key_pairs
