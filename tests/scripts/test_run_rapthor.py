"""Exercise the shell entry point without a Rapthor software installation."""

import os
import re
import shlex
import subprocess
import tarfile
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[2] / "sbin" / "run-rapthor.sh"


class WrapperEnvironment:
    def __init__(self, root):
        self.root = root
        self.work = self.root / "working directory"
        self.parset = self.root / "input.parset"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.install = self.root / "install"
        self.install.mkdir()
        (self.install / "sw-versions.txt").write_text("stub software versions\n")
        self.rc = self.root / "rapthor.rc"
        self.rc.write_text(
            f"export RAPTHOR_INSTALL_DIR={shlex.quote(str(self.install))}\n"
            f"export PATH={shlex.quote(str(self.bin))}:$PATH\n"
        )
        self.env = dict(os.environ, RAPTHOR_RC=str(self.rc), TEST_WORK=str(self.work))
        self.write_command("python", "exit 0\n")
        self.write_command(
            "rapthor",
            'printf "%s\\n" "$@" > "$TEST_ARGUMENTS"\n'
            'mkdir -p "$TEST_WORK/logs/nested"\n'
            'printf "pipeline log\\n" > "$TEST_WORK/logs/nested/run.log"\n'
            'printf "large science output\\n" > "$TEST_WORK/output.ms"\n'
            'exit "${TEST_STATUS:-0}"\n',
        )
        self.arguments = self.root / "arguments"
        self.env["TEST_ARGUMENTS"] = str(self.arguments)

    def write_command(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(0o755)

    def write_parset(self, extra=""):
        # Exercise whitespace around the delimiter and spaces in paths.
        self.parset.write_text(f"[global]\n  dir_working\t=\t{self.work}   \n" + extra)

    def run_wrapper(self, *args):
        return subprocess.run(
            ["bash", str(WRAPPER), *map(str, args)],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def assert_archive(self, expected):
        archives = list(self.work.glob("rapthor_logs_*.tar.gz"))
        assert len(archives) == 1
        assert re.fullmatch(r"rapthor_logs_\d{8}-\d{6}\.tar\.gz", archives[0].name)
        assert list(self.work.glob("tmp.*")) == []
        with tarfile.open(archives[0]) as archive:
            files = {member.name: member for member in archive if not member.isdir()}
            assert set(files) == set(expected)
            for name, content in expected.items():
                assert files[name].isfile(), name
                assert archive.extractfile(files[name]).read() == content

    def expected_archive(self):
        return {"./parset": self.parset.read_bytes(), "./logs/nested/run.log": b"pipeline log\n"}


@pytest.fixture
def wrapper(tmp_path):
    root = tmp_path / "run rapthor"
    root.mkdir()
    return WrapperEnvironment(root)


def test_archive_dereferences_inputs_and_preserves_existing_links(wrapper):
    wrapper.work.mkdir()
    inputs = {}
    for name in ("input_skymodel", "apparent_skymodel", "strategy"):
        path = wrapper.root / f"{name} file"
        path.write_text(f"{name} contents\n")
        inputs[name] = path
        (wrapper.work / path.name).symlink_to(wrapper.root / "missing old input")
    wrapper.write_parset("".join((f"  {name}\t= {path}\n" for name, path in inputs.items())))
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stdout + result.stderr
    assert wrapper.arguments.read_text().splitlines() == ["-v", str(wrapper.parset)]
    assert not (wrapper.work / wrapper.parset.name).exists()
    for path in inputs.values():
        assert (wrapper.work / path.name).is_symlink()
        assert (wrapper.work / path.name).readlink() == wrapper.root / "missing old input"
    wrapper.assert_archive(
        wrapper.expected_archive()
        | {f"./{name}": path.read_bytes() for name, path in inputs.items()}
    )


def test_existing_input_in_working_directory_is_preserved(wrapper):
    wrapper.work.mkdir()
    strategy = wrapper.work / "custom strategy.py"
    strategy.write_text("strategy contents\n")
    wrapper.write_parset(f"strategy = {strategy}\n")
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not strategy.is_symlink()
    wrapper.assert_archive(wrapper.expected_archive() | {"./strategy": strategy.read_bytes()})


def test_colon_delimited_paths_are_used_and_archived(wrapper):
    inputs = {}
    for name in ("input_skymodel", "apparent_skymodel", "strategy"):
        path = wrapper.root / f"{name}:file=value"
        path.write_text(f"{name} contents\n")
        inputs[name] = path
    wrapper.parset.write_text(
        f"[global]\n  dir_working\t:\t{wrapper.work}   \n"
        + "".join((f"  {name}\t: {path}   \n" for name, path in inputs.items()))
    )
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stdout + result.stderr
    assert wrapper.arguments.read_text().splitlines() == ["-v", str(wrapper.parset)]
    wrapper.assert_archive(
        wrapper.expected_archive()
        | {f"./{name}": path.read_bytes() for name, path in inputs.items()}
    )


@pytest.mark.parametrize(
    "extra", ("", "input_skymodel = None\napparent_skymodel = None\nstrategy = default\n")
)
def test_optional_inputs_can_be_absent_none_or_builtin(wrapper, extra):
    wrapper.write_parset(extra)
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stdout + result.stderr
    wrapper.assert_archive(wrapper.expected_archive())


def test_paths_in_other_sections_are_ignored(wrapper):
    inputs = {}
    for name in ("input_skymodel", "apparent_skymodel", "strategy"):
        path = wrapper.root / name
        path.write_text(f"{name} contents\n")
        inputs[name] = path
    misplaced = wrapper.root / "misplaced"
    misplaced.write_text("ignored\n")
    other_options = f"dir_working = {wrapper.root / 'wrong work'}\n" + "".join(
        (f"{name} = {misplaced}\n" for name in inputs)
    )
    wrapper.write_parset("".join((f"{name} = {path}\n" for name, path in inputs.items())))
    wrapper.parset.write_text(
        "[calibration]\n"
        + other_options
        + wrapper.parset.read_text()
        + "[imaging]\n"
        + other_options
    )
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARNING:" not in result.stderr
    assert not (wrapper.root / "wrong work").exists()
    wrapper.assert_archive(
        wrapper.expected_archive()
        | {f"./{name}": path.read_bytes() for name, path in inputs.items()}
    )


def test_missing_optional_file_warns_and_still_archives(wrapper):
    missing = wrapper.root / "missing skymodel"
    wrapper.write_parset(f"input_skymodel = {missing}\n")
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tar-ball may be incomplete" in result.stderr
    assert not (wrapper.work / missing.name).is_symlink()
    wrapper.assert_archive(wrapper.expected_archive())


def test_failure_status_is_preserved_after_archiving(wrapper):
    wrapper.write_parset()
    wrapper.env["TEST_STATUS"] = "37"
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 37, result.stdout + result.stderr
    assert "Rapthor execution failed, exit code: 37" in result.stderr
    assert "run-rapthor.sh exited with non-zero status: 37" in result.stderr
    wrapper.assert_archive(wrapper.expected_archive())


def test_archive_without_logs(wrapper):
    wrapper.write_command("rapthor", "exit 0\n")
    wrapper.write_parset()
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tar-ball may be incomplete" in result.stderr
    wrapper.assert_archive({"./parset": wrapper.parset.read_bytes()})


@pytest.mark.parametrize(
    "args, contents, message",
    [
        pytest.param([], "", "Usage:", id="no-arguments"),
        pytest.param(["input.parset", "input.parset"], "", "Usage:", id="extra-argument"),
        pytest.param(["missing.parset"], "", "not found", id="missing-parset"),
        pytest.param(
            ["input.parset"],
            "# dir_working = ignored\n",
            "Working directory invalid or not set",
            id="commented-working-directory",
        ),
        pytest.param(
            ["input.parset"],
            "dir_working =    \n",
            "Working directory invalid or not set",
            id="empty-working-directory",
        ),
        pytest.param(
            ["input.parset"],
            "dir_working = relative/path\n",
            "Working directory invalid or not set",
            id="relative-working-directory",
        ),
        pytest.param(
            ["input.parset"],
            "dir_working = None\n",
            "Working directory invalid or not set",
            id="none-working-directory",
        ),
        pytest.param(
            ["input.parset"],
            "[imaging]\ndir_working = {work}\n",
            "Working directory invalid or not set",
            id="wrong-section",
        ),
    ],
)
def test_invalid_arguments_and_working_directory_fail_before_rapthor(
    wrapper, args, contents, message
):
    args = [wrapper.root / arg for arg in args]
    contents = contents.format(work=wrapper.work)
    wrapper.parset.write_text("[global]\n" + contents)
    result = wrapper.run_wrapper(*args)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stderr
    assert "run-rapthor.sh exited with non-zero status: 1" in result.stderr
    assert "Rapthor exited" not in result.stderr
    assert not wrapper.arguments.exists()
    assert not wrapper.work.exists()


def test_relative_parset_is_resolved(wrapper):
    wrapper.write_parset()
    result = wrapper.run_wrapper(wrapper.parset.name)
    assert result.returncode == 0, result.stderr
    assert wrapper.arguments.read_text().splitlines() == ["-v", str(wrapper.parset)]
    wrapper.assert_archive(wrapper.expected_archive())


@pytest.mark.parametrize("status", (0, 37))
def test_archive_failure_preserves_rapthor_status_and_cleans_staging(wrapper, status):
    wrapper.write_command("tar", "exit 2\n")
    wrapper.write_parset()
    wrapper.env["TEST_STATUS"] = str(status)
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == status, result.stderr
    assert "tar-ball may be incomplete" in result.stderr
    assert list(wrapper.work.glob("tmp.*")) == []


def test_setup_failure_cleans_staging(wrapper):
    wrapper.write_command("python", "exit 23\n")
    wrapper.write_parset()
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 23, result.stderr
    assert "run-rapthor.sh exited with non-zero status: 23" in result.stderr
    assert "Rapthor exited" not in result.stderr
    assert not wrapper.arguments.exists()
    assert list(wrapper.work.iterdir()) == []


def test_missing_environment_fails_before_rapthor(wrapper):
    wrapper.env["RAPTHOR_RC"] = str(wrapper.root / "missing.rc")
    wrapper.write_parset()
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode != 0
    assert "Failed to source rapthor environment" in result.stderr
    assert "run-rapthor.sh exited with non-zero status:" in result.stderr
    assert "Rapthor exited" not in result.stderr
    assert not wrapper.arguments.exists()
    assert not wrapper.work.exists()


def test_inputs_with_same_basename_are_archived_separately(wrapper):
    inputs = {}
    for name in ("input_skymodel", "apparent_skymodel", "strategy"):
        directory = wrapper.root / name
        directory.mkdir()
        path = directory / "input"
        path.write_text(f"{name} contents\n")
        inputs[name] = path
    wrapper.write_parset("".join((f"{name} = {path}\n" for name, path in inputs.items())))
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stderr
    wrapper.assert_archive(
        wrapper.expected_archive()
        | {f"./{name}": path.read_bytes() for name, path in inputs.items()}
    )


@pytest.mark.parametrize("quote", ("'", '"'))
def test_quoted_absolute_paths(wrapper, quote):
    inputs = {}
    for name in ("input_skymodel", "apparent_skymodel", "strategy"):
        path = wrapper.root / f"{name} file"
        path.write_text(f"{name} contents\n")
        inputs[name] = path
    wrapper.parset.write_text(
        f"[global]\n  dir_working = {quote}{wrapper.work}{quote}   \n"
        + "".join((f"  {name} = {quote}{path}{quote}   \n" for name, path in inputs.items()))
    )
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stderr
    assert "WARNING:" not in result.stderr
    wrapper.assert_archive(
        wrapper.expected_archive()
        | {f"./{name}": path.read_bytes() for name, path in inputs.items()}
    )
