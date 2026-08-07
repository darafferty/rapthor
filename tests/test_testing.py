"""Tests for helpers in ``rapthor.testing``."""

import pytest

import rapthor.testing as testing


@pytest.mark.parametrize(("available_cpus", "expected_cpus"), [(4, 4), (12, 6)])
def test_generate_parset_from_template_caps_cpu_requests(
    tmp_path, monkeypatch, available_cpus, expected_cpus
):
    template_path = tmp_path / "template.parset"
    template_path.write_text("[cluster]\n", encoding="utf-8")
    monkeypatch.setattr(testing.misc, "nproc", lambda: available_cpus)
    monkeypatch.setenv("CI_PROJECT_DIR", str(tmp_path))

    parset = testing.generate_parset_from_template(
        template_path,
        tmp_path / "input.ms",
        cpu_limit=6,
    )

    for option in ("cpus_per_task", "max_cores", "max_threads"):
        assert parset.getint("cluster", option) == expected_cpus


def test_generate_parset_from_template_rejects_invalid_cpu_limit(tmp_path):
    with pytest.raises(ValueError, match="cpu_limit must be at least 1"):
        testing.generate_parset_from_template(
            tmp_path / "template.parset",
            tmp_path / "input.ms",
            cpu_limit=0,
        )
