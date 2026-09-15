from tsdm.losses import SequenceLoss


class TestSequenceLoss:
    loss: SequenceLoss

    def test_univariate_series(self):
        # ensures that the case of univariate series of shape Float[..., $N]
        # is properly handled. in this case channel_dim=(), but there are some
        # footguns, e.g. torch.sum(dim=()) reduces all dimensions rather than none.
        pass

    def test_backprop_predictions(self):
        # ensure that gradients w.r.t. predictions propagate, even when targets are
        # contaminated with NaNs.
        pass
