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
    assert parser["cluster"]["prefect_command_profile"] == "perf"
    assert "cwl_runner" not in parser["cluster"]


def test_generated_rich_strategy_exercises_amplitude_solves_with_flux_threshold(tmp_path):
    module = load_generator_script()
    strategy_path = tmp_path / "prefect_demo_rich_strategy.py"

    module.write_rich_strategy(strategy_path)

    spec = importlib.util.spec_from_file_location("prefect_demo_rich_strategy", strategy_path)
    strategy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = strategy
    spec.loader.exec_module(strategy)

    assert any(step["do_slowgain_solve"] for step in strategy.strategy_steps)
    assert any(step["do_fulljones_solve"] for step in strategy.strategy_steps)
    assert all(step["target_flux"] is not None for step in strategy.strategy_steps)
    assert all(step["target_flux"] > 0.1 for step in strategy.strategy_steps)


def test_checked_in_demo_parset_uses_prefect_settings_only():
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(REPO_ROOT / "examples" / "prefect_demo.parset")

    assert parser["cluster"]["prefect_task_runner"] == "local_dask"
    assert parser["cluster"]["prefect_command_profile"] == "perf"
    assert "cwl_runner" not in parser["cluster"]
