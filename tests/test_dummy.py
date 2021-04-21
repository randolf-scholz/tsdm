import tsdm


def dummy_test():
    assert True


def check_available_models():
    assert 'M3' in tsdm.AVAILABLE_MODELS
