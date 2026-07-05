import configparser
import importlib.util
import sys
from pathlib import Path

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


def test_generated_benchmark_parset_uses_runtime_sized_thread_defaults(tmp_path):
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
    assert parser["cluster"]["prefect_task_runner"] == "local_dask"
    assert parser["cluster"]["local_dask_workers"] == "2"
    assert parser["cluster"]["cpus_per_task"] == "0"
    assert parser["cluster"]["max_cores"] == "0"
    assert parser["cluster"]["max_threads"] == "0"
    assert parser["cluster"]["deconvolution_threads"] == "0"
    assert parser["cluster"]["parallel_gridding_tasks"] == "0"


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


def test_checked_in_demo_parset_uses_prefect_settings_only():
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(REPO_ROOT / "examples" / "prefect_demo.parset")

    assert parser["cluster"]["prefect_task_runner"] == "local_dask"
    assert parser["cluster"]["max_nodes"] == "1"
    assert parser["cluster"]["local_dask_workers"] == "1"
    assert parser["cluster"]["prefect_command_profile"] == "time"
    assert "cwl_runner" not in parser["cluster"]
