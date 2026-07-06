"""Tests for fast shared integration-test defaults."""

from tests.conftest import generate_parset
from tests.integration.conftest import make_strategy_step


def test_generated_integration_parset_uses_fast_shared_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPTHOR_TEST_RUN_ROOT", str(tmp_path / "runs"))

    parset = generate_parset(
        "tests/resources/integration_template.parset",
        tmp_path / "input.ms",
    )

    assert parset["imaging"]["grid_width_ra_deg"] == "0.25"
    assert parset["imaging"]["grid_width_dec_deg"] == "0.25"
    assert parset["cluster"]["cpus_per_task"] == "6"
    assert parset["cluster"]["max_cores"] == "6"
    assert parset["cluster"]["max_threads"] == "6"
    assert parset["cluster"]["deconvolution_threads"] == "3"
    assert parset["cluster"]["parallel_gridding_tasks"] == "3"


def test_make_strategy_step_uses_fast_shared_imaging_depth():
    step = make_strategy_step(do_calibrate=True)

    assert step["auto_mask"] == 7.0
    assert step["auto_mask_nmiter"] == 1
    assert step["max_nmiter"] == 2
    assert step["do_calibrate"] is True


def test_make_strategy_step_keeps_explicit_test_overrides():
    step = make_strategy_step(max_nmiter=8)

    assert step["max_nmiter"] == 8
