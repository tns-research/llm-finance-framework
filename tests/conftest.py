"""Shared pytest configuration.

Tests marked ``@pytest.mark.network`` hit live external services (e.g. Stooq) and
are skipped by default so the suite runs offline and deterministically. Pass
``--run-network`` to opt in.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.network (live internet access)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network option to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
