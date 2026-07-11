import configparser
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from rapthor.lib.field import Field
from rapthor.lib.parset import parset_read

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "generate-prefect-demo-data.py"
REPO_ROOT = Path(__file__).parents[2]


def load_generator_script():
    spec = importlib.util.spec_from_file_location("generate_prefect_demo_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_demo_parset_uses_prefect_settings_only(tmp_path):
    module = load_generator_script()
    output_dir = tmp_path / "generated" / "prefect_demo_rich"
    output_dir.mkdir(parents=True)
    parset_path = output_dir / "prefect_demo_rich.parset"

    module.write_parset(
        output_dir,
        parset_path,
        REPO_ROOT,
        REPO_ROOT / "examples" / "prefect_demo_strategy.py",
    )

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(parset_path)
    assert parser["cluster"]["prefect_task_runner"] == "local_dask"
    assert parser["cluster"]["max_nodes"] == "1"
    assert parser["cluster"]["local_dask_workers"] == "2"
    assert parser["cluster"]["prefect_command_profile"] == "time"
    assert parser["cluster"]["prefect_run_tags"] == "prefect-demo,rich-demo"
    assert parser["cluster"]["prefect_publish_fits_previews"] == "True"
    assert parser["cluster"]["prefect_publish_postage_stamp_previews"] == "True"
    assert parser["cluster"]["prefect_postage_stamp_preview_count"] == "5"
    assert parser["cluster"]["prefect_postage_stamp_preview_size_px"] == "96"
    assert parser["cluster"]["prefect_fits_preview_clip_percentile"] == "99.9"
    assert parser["imaging"]["dde_method"] == "full"
    assert "cwl_runner" not in parser["cluster"]


def test_generated_demo_parset_can_use_benchmark_strategy(tmp_path):
    module = load_generator_script()
    output_dir = tmp_path / "generated" / "prefect_demo_rich"
    output_dir.mkdir(parents=True)
    parset_path = output_dir / "prefect_demo_rich.parset"
    strategy_path = output_dir / "prefect_demo_benchmark_strategy.py"

    module.write_benchmark_strategy(strategy_path)
    module.write_parset(output_dir, parset_path, REPO_ROOT, strategy_path)

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(parset_path)
    assert parser["global"]["strategy"] == str(strategy_path)
    assert parser["global"]["dir_working"].endswith("work")
    assert parser["cluster"]["cpus_per_task"] == "4"
    assert parser["cluster"]["max_threads"] == "4"
    assert parser["cluster"]["prefect_publish_fits_previews"] == "True"
    assert parser["cluster"]["prefect_publish_postage_stamp_previews"] == "True"


def test_generated_benchmark_parset_uses_runtime_sized_threads_and_filter_default(tmp_path):
    module = load_generator_script()
    output_dir = tmp_path / "generated" / "prefect_demo_rich"
    output_dir.mkdir(parents=True)
    parset_path = output_dir / "prefect_demo_benchmark.parset"
    strategy_path = output_dir / "prefect_demo_benchmark_strategy.py"

    module.write_benchmark_parset(output_dir, parset_path, REPO_ROOT, strategy_path)

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(parset_path)
    assert parser["global"]["strategy"] == str(strategy_path)
    assert parser["global"]["dir_working"].endswith("benchmark-work")
    assert parser["imaging"]["grid_width_ra_deg"] == "1.0"
    assert parser["imaging"]["grid_width_dec_deg"] == "1.0"
    assert parser["imaging"]["dde_method"] == "full"
    assert parser["cluster"]["prefect_task_runner"] == "local_dask"
    assert parser["cluster"]["local_dask_workers"] == "2"
    assert parser["cluster"]["cpus_per_task"] == "0"
    assert parser["cluster"]["max_cores"] == "0"
    assert parser["cluster"]["max_threads"] == "0"
    assert parser["cluster"]["filter_skymodel_ncores"] == "15"
    assert parser["cluster"]["deconvolution_threads"] == "0"
    assert parser["cluster"]["parallel_gridding_tasks"] == "0"
    assert parser["cluster"]["prefect_run_tags"] == "prefect-demo,CI-benchmark"
    assert parser["cluster"]["prefect_publish_fits_previews"] == "False"
    assert parser["cluster"]["prefect_publish_postage_stamp_previews"] == "False"
    assert parser["cluster"]["prefect_postage_stamp_preview_count"] == "5"
    assert parser["cluster"]["prefect_postage_stamp_preview_size_px"] == "96"
    assert parser["cluster"]["prefect_fits_preview_clip_percentile"] == "99.9"


def test_generated_normalization_reference_skymodels_use_distinct_frequencies(tmp_path):
    module = load_generator_script()
    output_dir = tmp_path / "generated" / "prefect_demo_rich"
    output_dir.mkdir(parents=True)
    low_path = output_dir / "reference_low.txt"
    high_path = output_dir / "reference_high.txt"
    source = module.SOURCES[0]

    module.write_sky_model(
        low_path,
        apparent=False,
        reference_frequency_hz=120000000.0,
        flux_frequency_hz=120000000.0,
    )
    module.write_sky_model(
        high_path,
        apparent=False,
        reference_frequency_hz=160000000.0,
        flux_frequency_hz=160000000.0,
    )

    low_text = low_path.read_text(encoding="utf-8")
    high_text = high_path.read_text(encoding="utf-8")
    low_source_line = next(line for line in low_text.splitlines() if line.startswith(source.name))
    high_source_line = next(line for line in high_text.splitlines() if line.startswith(source.name))
    low_flux = float(low_source_line.split(",")[5])
    high_flux = float(high_source_line.split(",")[5])

    assert "ReferenceFrequency='120000000.0'" in low_text
    assert "ReferenceFrequency='160000000.0'" in high_text
    assert low_flux == pytest.approx(
        source.true_flux_jy
        * (120000000.0 / module.REFERENCE_FREQUENCY_HZ) ** source.spectral_index,
        rel=1e-5,
    )
    assert high_flux == pytest.approx(
        source.true_flux_jy
        * (160000000.0 / module.REFERENCE_FREQUENCY_HZ) ** source.spectral_index,
        rel=1e-5,
    )
    assert low_flux != pytest.approx(high_flux)


def test_multi_sector_layout_places_bright_patches_inside_all_quadrants():
    module = load_generator_script()

    patch_quadrants = {
        (patch.delta_ra_deg > 0, patch.delta_dec_deg > 0) for patch in module.MULTI_SECTOR_PATCHES
    }
    source_patch_names = {source.patch for source in module.MULTI_SECTOR_SOURCES}

    assert patch_quadrants == {
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    }
    assert all(0.25 <= abs(patch.delta_ra_deg) <= 0.35 for patch in module.MULTI_SECTOR_PATCHES)
    assert all(0.25 <= abs(patch.delta_dec_deg) <= 0.35 for patch in module.MULTI_SECTOR_PATCHES)
    assert source_patch_names == {patch.name for patch in module.MULTI_SECTOR_PATCHES}


def test_generated_multi_sector_benchmark_parset_uses_quadrant_dataset_and_grid(tmp_path):
    module = load_generator_script()
    output_dir = tmp_path / "generated" / "prefect_demo_rich"
    output_dir.mkdir(parents=True)
    parset_path = output_dir / "prefect_demo_multisector_benchmark.parset"
    strategy_path = output_dir / "prefect_demo_benchmark_strategy.py"

    module.write_benchmark_strategy(strategy_path)
    module.write_multi_sector_benchmark_parset(output_dir, parset_path, REPO_ROOT, strategy_path)

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(parset_path)
    assert parser["global"]["input_ms"].endswith("prefect_demo_multisector.ms")
    assert parser["global"]["input_skymodel"].endswith("prefect_demo_multisector_true_sky.txt")
    assert parser["global"]["apparent_skymodel"].endswith(
        "prefect_demo_multisector_apparent_sky.txt"
    )
    assert parser["global"]["dir_working"].endswith("multisector-benchmark-work")
    assert parser["cluster"]["prefect_run_tags"] == "prefect-demo,CI-multi-sector-benchmark"
    assert parser["imaging"]["grid_width_ra_deg"] == "1.5"
    assert parser["imaging"]["grid_width_dec_deg"] == "1.5"
    assert parser["imaging"]["grid_nsectors_ra"] == "2"
    assert parser["imaging"]["dde_method"] == "single"
    assert parser["imaging"]["skip_corner_sectors"] == "False"
    assert parser["cluster"]["filter_skymodel_ncores"] == "15"
    assert parser["cluster"]["prefect_publish_fits_previews"] == "False"


def test_generated_multi_sector_benchmark_parset_builds_four_grid_sectors(tmp_path, test_ms):
    module = load_generator_script()
    output_dir = tmp_path / "generated" / "prefect_demo_rich"
    output_dir.mkdir(parents=True)
    shutil.copytree(test_ms, output_dir / "prefect_demo_multisector.ms")
    module.write_sky_model(
        output_dir / "prefect_demo_multisector_true_sky.txt",
        apparent=False,
        patches=module.MULTI_SECTOR_PATCHES,
        sources=module.MULTI_SECTOR_SOURCES,
    )
    module.write_sky_model(
        output_dir / "prefect_demo_multisector_apparent_sky.txt",
        apparent=True,
        patches=module.MULTI_SECTOR_PATCHES,
        sources=module.MULTI_SECTOR_SOURCES,
    )
    strategy_path = output_dir / "prefect_demo_benchmark_strategy.py"
    parset_path = output_dir / "prefect_demo_multisector_benchmark.parset"

    module.write_benchmark_strategy(strategy_path)
    module.write_multi_sector_benchmark_parset(output_dir, parset_path, REPO_ROOT, strategy_path)

    field = Field(parset_read(parset_path, use_log_file=False))

    assert field.uses_sector_grid is True
    assert len(field.imaging_sectors) == 4
    assert [sector.name for sector in field.imaging_sectors] == [
        "sector_1",
        "sector_2",
        "sector_3",
        "sector_4",
    ]
    assert all(sector.width_ra == pytest.approx(0.75) for sector in field.imaging_sectors)
    assert all(sector.width_dec == pytest.approx(0.75) for sector in field.imaging_sectors)


def test_generated_benchmark_strategy_exercises_legacy_default_and_fulljones(tmp_path):
    module = load_generator_script()
    strategy_path = tmp_path / "prefect_demo_benchmark_strategy.py"

    module.write_benchmark_strategy(strategy_path)

    spec = importlib.util.spec_from_file_location("prefect_demo_benchmark_strategy", strategy_path)
    strategy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = strategy
    spec.loader.exec_module(strategy)

    assert len(strategy.strategy_steps) == 4
    assert strategy.strategy_steps[0]["calibration_strategy"] == {
        "di": ["fast_phase", "medium_phase"]
    }
    assert strategy.strategy_steps[1]["calibration_strategy"] == {
        "di": [],
        "dd": ["fast_phase", "medium_phase"],
    }
    assert strategy.strategy_steps[1]["regroup_model"] is True
    assert strategy.strategy_steps[2]["calibration_strategy"] == {
        "di": [],
        "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
    }
    assert strategy.strategy_steps[2]["regroup_model"] is True
    assert strategy.strategy_steps[3]["calibration_strategy"] == {
        "di": ["full_jones"],
        "dd": [],
    }
    assert strategy.strategy_steps[3]["regroup_model"] is False
    assert all(step["target_flux"] >= 0.6 for step in strategy.strategy_steps)


def test_generated_benchmark_normalize_strategy_enables_first_cycle_normalization(tmp_path):
    module = load_generator_script()
    strategy_path = tmp_path / "prefect_demo_benchmark_normalize_strategy.py"

    module.write_benchmark_normalize_strategy(strategy_path)

    spec = importlib.util.spec_from_file_location(
        "prefect_demo_benchmark_normalize_strategy", strategy_path
    )
    strategy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = strategy
    spec.loader.exec_module(strategy)

    assert strategy.strategy_steps[0]["do_normalize"] is True
    assert all(step["do_normalize"] is False for step in strategy.strategy_steps[1:])


def test_generated_calibration_postprocess_strategy_is_calibration_only(tmp_path):
    module = load_generator_script()
    strategy_path = tmp_path / "prefect_demo_benchmark_calibration_postprocess_strategy.py"

    module.write_benchmark_calibration_postprocess_strategy(strategy_path)

    spec = importlib.util.spec_from_file_location(
        "prefect_demo_benchmark_calibration_postprocess_strategy", strategy_path
    )
    strategy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = strategy
    spec.loader.exec_module(strategy)

    assert len(strategy.strategy_steps) == 1
    step = strategy.strategy_steps[0]
    assert step["do_calibrate"] is True
    assert step["do_image"] is False
    assert step["calibration_strategy"] == {
        "di": [],
        "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"],
    }
    assert step["regroup_model"] is True


def test_checked_in_demo_parset_uses_prefect_settings_only():
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(REPO_ROOT / "examples" / "prefect_demo.parset")

    assert parser["cluster"]["prefect_task_runner"] == "local_dask"
    assert parser["cluster"]["max_nodes"] == "1"
    assert parser["cluster"]["local_dask_workers"] == "1"
    assert parser["cluster"]["prefect_command_profile"] == "time"
    assert parser["cluster"]["prefect_run_tags"] == "prefect-demo,simple-demo"
    assert parser["cluster"]["prefect_publish_fits_previews"] == "True"
    assert parser["cluster"]["prefect_publish_postage_stamp_previews"] == "True"
    assert parser["cluster"]["prefect_postage_stamp_preview_count"] == "5"
    assert parser["cluster"]["prefect_postage_stamp_preview_size_px"] == "96"
    assert parser["cluster"]["prefect_fits_preview_clip_percentile"] == "99.9"
    assert "cwl_runner" not in parser["cluster"]
