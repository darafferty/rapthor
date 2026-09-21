"""Contracts for explicit external-command environment policies."""

import os

import pytest

from rapthor.execution.environments import (
    dp3_environment,
    filter_skymodel_environment,
    thread_environment,
    wsclean_environment,
)
from rapthor.execution.resources import ResourceRequest


@pytest.mark.parametrize("inherited_trim", [None, "65536"])
def test_dp3_environment_only_overrides_child_allocator(monkeypatch, inherited_trim):
    if inherited_trim is None:
        monkeypatch.delenv("MALLOC_TRIM_THRESHOLD_", raising=False)
    else:
        monkeypatch.setenv("MALLOC_TRIM_THRESHOLD_", inherited_trim)
    original_environment = dict(os.environ)

    environment = dp3_environment()

    assert environment == {"MALLOC_TRIM_THRESHOLD_": None}
    assert dict(os.environ) == original_environment
    assert dp3_environment() is not environment


def test_thread_environment_sets_common_thread_variables():
    assert thread_environment(ResourceRequest(threads=3)) == {
        "OMP_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
    }


@pytest.mark.parametrize("threads", [1, 6])
@pytest.mark.parametrize("use_mpi", [False, True])
@pytest.mark.parametrize("inherited_trim", [None, "65536"])
def test_wsclean_environment_preserves_threads_and_isolates_allocator(
    monkeypatch, threads, use_mpi, inherited_trim
):
    if inherited_trim is None:
        monkeypatch.delenv("MALLOC_TRIM_THRESHOLD_", raising=False)
    else:
        monkeypatch.setenv("MALLOC_TRIM_THRESHOLD_", inherited_trim)
    original_environment = dict(os.environ)
    request = ResourceRequest(threads=threads, use_mpi=use_mpi)

    environment = wsclean_environment(request)

    expected = {"MALLOC_TRIM_THRESHOLD_": None}
    if use_mpi:
        expected.update({"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1"})
    else:
        expected["DUCC0_NUM_THREADS"] = str(threads)
    assert environment == expected
    assert dict(os.environ) == original_environment
    assert wsclean_environment(request) is not environment


@pytest.mark.parametrize("inherited_trim", [None, "65536"])
def test_filter_skymodel_environment_isolates_allocator_and_limits_native_threads(
    monkeypatch, inherited_trim
):
    if inherited_trim is None:
        monkeypatch.delenv("MALLOC_TRIM_THRESHOLD_", raising=False)
    else:
        monkeypatch.setenv("MALLOC_TRIM_THRESHOLD_", inherited_trim)
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    for variable in thread_variables:
        monkeypatch.setenv(variable, "192")
    original_environment = dict(os.environ)

    environment = filter_skymodel_environment()

    assert environment == {
        "MALLOC_TRIM_THRESHOLD_": None,
        **{variable: "1" for variable in thread_variables},
    }
    assert dict(os.environ) == original_environment
    assert filter_skymodel_environment() is not environment
