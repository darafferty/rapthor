import configparser
import json
from pathlib import Path

import pytest

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


def _read_materialized(output_file):
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(output_file)
    return parser


def test_materialize_parset_paths_reads_json_values(tmp_path):
    parset_file = tmp_path / "input.JSON"
    output_file = tmp_path / "input.materialized.parset"
    parset_file.write_text(
        json.dumps(
            {
                "global": {
                    "dir_working": "work",
                    "input_ms": ["a.ms", "b.ms"],
                    "input_skymodel": None,
                    "regroup_input_skymodel": True,
                    "selfcal_data_fraction": 0.2,
                },
                "imaging": {"dde_method": "single", "cellsize_arcsec": 1.5},
            }
        ),
        encoding="utf-8",
    )

    materialize_parset_paths(parset_file, output_file, base_dir=tmp_path)

    parser = _read_materialized(output_file)
    assert parser["global"]["dir_working"] == str(tmp_path / "work")
    assert parser["global"]["input_ms"] == f"[{tmp_path / 'a.ms'}, {tmp_path / 'b.ms'}]"
    assert parser["global"]["input_skymodel"] == "None"
    assert parser["global"]["regroup_input_skymodel"] == "True"
    assert parser["global"]["selfcal_data_fraction"] == "0.2"
    assert parser["imaging"]["dde_method"] == "single"
    assert parser["imaging"]["cellsize_arcsec"] == "1.5"


@pytest.mark.parametrize(
    "content, message",
    [
        ("{not json", "Expecting property name"),
        ("[1, 2]", "top-level entry must be a dictionary"),
        ('{"global": 1}', "secondary-level entries must be dictionaries"),
        ('{"global": {"strategy": "a", "strategy": "b"}}', "Duplicate key 'strategy'"),
        ('{"global": {"input_ms": "a", "Input_MS": "b"}}', "already exists"),
        ('{"global": {"input_ms": {"a": 1}}}', "must be a scalar or a list, not an object"),
        ('{"global": {"input_ms": [["a"]]}}', "List items of option 'input_ms'"),
        ('{"global": {"input_ms": [{"a": 1}]}}', "must be scalars"),
        ('{"global": {"input_ms": ["a,b.ms"]}}', "must not contain commas"),
    ],
)
def test_materialize_parset_paths_rejects_invalid_json(tmp_path, content, message):
    parset_file = tmp_path / "input.json"
    parset_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="JSON parset could not be parsed correctly") as exc:
        materialize_parset_paths(parset_file, tmp_path / "out.parset", base_dir=tmp_path)

    assert message in str(exc.value)
