import configparser
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "dev" / "generate-prefect-demo-data.py"


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
    repo_root = Path.cwd()

    module.write_parset(
        output_dir,
        parset_path,
        repo_root,
        repo_root / "examples" / "prefect_demo_strategy.py",
    )

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(parset_path)
    assert parser["cluster"]["prefect_task_runner"] == "local_dask"
    assert "cwl_runner" not in parser["cluster"]


def test_checked_in_demo_parset_uses_prefect_settings_only():
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(Path.cwd() / "examples" / "prefect_demo.parset")

    assert parser["cluster"]["prefect_task_runner"] == "local_dask"
    assert "cwl_runner" not in parser["cluster"]
