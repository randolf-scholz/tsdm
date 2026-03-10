from tsdm.datatools.folds import folds_as_frame


def test_folds_as_frame() -> None:
    r"""Test `folds_as_frame`."""
    folds = [
        {"train": [0, 1, 2], "valid": [3], "test": [4]},
        {"train": [1, 2, 3], "valid": [4], "test": [0]},
    ]
    expected = {
        0: {0: "train", 1: "train", 2: "train", 3: "valid", 4: "test"},
        1: {0: "test", 1: "train", 2: "train", 3: "train", 4: "valid"},
    }

    df = folds_as_frame(folds)
    assert df.shape == (5, 2)
    assert df.to_dict() == expected


def test_folds_as_frame_mask() -> None:
    r"""Test `folds_as_frame`."""
    folds = [
        {
            "train": [True, True, True, False, False],
            "valid": [False, False, False, True, False],
            "test": [False, False, False, False, True],
        },
        {
            "train": [False, True, True, True, False],
            "valid": [False, False, False, False, True],
            "test": [True, False, False, False, False],
        },
    ]
    expected = {
        0: {0: "train", 1: "train", 2: "train", 3: "valid", 4: "test"},
        1: {0: "test", 1: "train", 2: "train", 3: "train", 4: "valid"},
    }

    df = folds_as_frame(folds)
    assert df.shape == (5, 2)
    assert df.to_dict() == expected


def test_fold_as_frame_sparse() -> None:
    r"""Test `folds_as_frame` with `sparse=True`."""
    folds = [
        {"train": [0, 1, 2], "valid": [3], "test": [4]},
        {"train": [1, 2, 3], "valid": [4], "test": [0]},
    ]
    expected = {
        (0, "test"): {0: False, 1: False, 2: False, 3: False, 4: True},
        (0, "train"): {0: True, 1: True, 2: True, 3: False, 4: False},
        (0, "valid"): {0: False, 1: False, 2: False, 3: True, 4: False},
        (1, "test"): {0: True, 1: False, 2: False, 3: False, 4: False},
        (1, "train"): {0: False, 1: True, 2: True, 3: True, 4: False},
        (1, "valid"): {0: False, 1: False, 2: False, 3: False, 4: True},
    }
    df = folds_as_frame(folds, sparse=True)
    assert df.shape == (5, len(folds) * 3)
    assert df.to_dict() == expected


def test_fold_as_frame_sparse_mask() -> None:
    r"""Test `folds_as_frame` with `sparse=True`."""
    folds = [
        {
            "train": [True, True, True, False, False],
            "valid": [False, False, False, True, False],
            "test": [False, False, False, False, True],
        },
        {
            "train": [False, True, True, True, False],
            "valid": [False, False, False, False, True],
            "test": [True, False, False, False, False],
        },
    ]
    expected = {
        (0, "test"): {0: False, 1: False, 2: False, 3: False, 4: True},
        (0, "train"): {0: True, 1: True, 2: True, 3: False, 4: False},
        (0, "valid"): {0: False, 1: False, 2: False, 3: True, 4: False},
        (1, "test"): {0: True, 1: False, 2: False, 3: False, 4: False},
        (1, "train"): {0: False, 1: True, 2: True, 3: True, 4: False},
        (1, "valid"): {0: False, 1: False, 2: False, 3: False, 4: True},
    }
    df = folds_as_frame(folds, sparse=True)
    assert df.shape == (5, len(folds) * 3)
    assert df.to_dict() == expected
