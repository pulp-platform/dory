import pytest


def pytest_addoption(parser):
    parser.addoption("--compat",
                     action="store",
                     default="gap-sdk",
                     type=str,
                     choices=("gap-sdk", "pulp-sdk"))
    parser.addoption("--appdir",
                     action="store",
                     default=None)

@pytest.fixture
def compat(request):
    return request.config.getoption("--compat")

@pytest.fixture
def appdir(request):
    return request.config.getoption("--appdir")
