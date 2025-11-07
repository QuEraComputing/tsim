import tsim


def test_version():
    assert hasattr(tsim, "__version__")
    assert isinstance(tsim.__version__, str)
