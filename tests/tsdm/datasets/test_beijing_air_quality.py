from tsdm.datasets import BeijingAirQuality


def test_beijing_air_quality():
    BeijingAirQuality.reset_dataset_files(force=True)
    ds = BeijingAirQuality()
