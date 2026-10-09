import configparser
from pathlib import Path

from rapthor.lib.parset_paths import materialize_parset_paths, resolve_path_value


def test_resolve_path_value_keeps_empty_values():
    assert resolve_path_value("", Path.cwd()) == ""
    assert resolve_path_value("None", Path.cwd()) == "None"


def test_resolve_path_value_keeps_urls():
    assert resolve_path_value("https://example.test/catalog", Path.cwd()) == (
        "https://example.test/catalog"
    )


def test_resolve_path_value_resolves_list_tokens(tmp_path):
    resolved = resolve_path_value("[a.txt, b.txt]", tmp_path)

    assert resolved == f"[{tmp_path / 'a.txt'}, {tmp_path / 'b.txt'}]"


def test_materialize_parset_paths_resolves_known_path_options(tmp_path):
    parset_file = tmp_path / "input.parset"
    output_file = tmp_path / "runtime" / "input.materialized.parset"
    parset_file.write_text(
        "\n".join(
            [
                "[global]",
                "dir_working = work",
                "input_ms = data/input.ms",
                "input_skymodel = sky/true.txt",
                "strategy = strategies/demo.py",
                "",
                "[imaging]",
                "normalization_skymodels = [sky/a.txt, sky/b.txt]",
            ]
        ),
        encoding="utf-8",
    )

    materialize_parset_paths(parset_file, output_file, base_dir=tmp_path)

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(output_file)
    assert parser["global"]["dir_working"] == str(tmp_path / "work")
    assert parser["global"]["input_ms"] == str(tmp_path / "data/input.ms")
    assert parser["global"]["input_skymodel"] == str(tmp_path / "sky/true.txt")
    assert parser["global"]["strategy"] == str(tmp_path / "strategies/demo.py")
    assert parser["imaging"]["normalization_skymodels"] == (
        f"[{tmp_path / 'sky/a.txt'}, {tmp_path / 'sky/b.txt'}]"
    )


def test_materialize_parset_paths_can_override_working_dir(tmp_path):
    parset_file = tmp_path / "input.parset"
    output_file = tmp_path / "input.materialized.parset"
    override = tmp_path / "override-work"
    parset_file.write_text("[global]\ndir_working = work\n", encoding="utf-8")

    materialize_parset_paths(
        parset_file,
        output_file,
        working_dir_override=override,
        base_dir=tmp_path,
    )

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(output_file)
    assert parser["global"]["dir_working"] == str(override)
