import pytest


def pytest_addoption(parser):
    parser.addoption("--compat",
                     action="store",
                     default="gap-sdk",
                     type=str,
                     choices=("gap-sdk", "pulp-sdk"))

@pytest.fixture
def compat(request):
    return request.config.getoption("--compat")
