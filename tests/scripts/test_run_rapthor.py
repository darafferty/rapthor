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

    def assert_archive(self, extra=None, logs=True):
        expected = {"./parset": self.parset.read_bytes(), **(extra or {})}
        if logs:
            expected["./logs/nested/run.log"] = b"pipeline log\n"
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


@pytest.fixture
def wrapper(tmp_path):
    root = tmp_path / "run rapthor"
    root.mkdir()
    return WrapperEnvironment(root)


@pytest.mark.parametrize("delimiter, quote", [("=", ""), (":", ""), ("=", "'"), (":", '"')])
def test_archive_inputs(wrapper, delimiter, quote):
    wrapper.work.mkdir()
    inputs = {}
    for name in ("input_skymodel", "apparent_skymodel", "strategy"):
        directory = wrapper.work / name
        directory.mkdir()
        path = directory / "same input:name=value"
        path.write_text(f"{name} contents\n")
        inputs[name] = path
    old_link = wrapper.work / path.name
    old_link.symlink_to(wrapper.root / "missing old input")
    options = {"dir_working": wrapper.work, **inputs}
    ignored_options = "".join(f"{name} = /ignored\n" for name in options)
    wrapper.parset.write_text(
        "[calibration]\n"
        + ignored_options
        + "[global]\n"
        + "".join(
            f"  {name}\t{delimiter}\t{quote}{path}{quote}   \n" for name, path in options.items()
        )
        + "[imaging]\n"
        + ignored_options
    )
    result = wrapper.run_wrapper(wrapper.parset.name)
    assert result.returncode == 0, result.stderr
    assert "WARNING:" not in result.stderr
    assert wrapper.arguments.read_text().splitlines() == ["-v", str(wrapper.parset)]
    assert not (wrapper.work / wrapper.parset.name).exists()
    assert old_link.readlink() == wrapper.root / "missing old input"
    for path in inputs.values():
        assert not path.is_symlink()
    wrapper.assert_archive({f"./{name}": path.read_bytes() for name, path in inputs.items()})


@pytest.mark.parametrize(
    "extra", ["", "input_skymodel = None\napparent_skymodel = None\nstrategy = default\n"]
)
def test_optional_inputs(wrapper, extra):
    wrapper.write_parset(extra)
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stderr
    wrapper.assert_archive()


@pytest.mark.parametrize("missing", ["input", "logs"])
def test_missing_archive_files_warn(wrapper, missing):
    extra = f"input_skymodel = {wrapper.root / 'missing'}\n" if missing == "input" else ""
    if missing == "logs":
        wrapper.write_command("rapthor", "exit 0\n")
    wrapper.write_parset(extra)
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 0, result.stderr
    assert "tar-ball may be incomplete" in result.stderr
    wrapper.assert_archive(logs=missing != "logs")


@pytest.mark.parametrize("status", [0, 37])
@pytest.mark.parametrize("archive_fails", [False, True])
def test_exit_status_and_cleanup(wrapper, status, archive_fails):
    if archive_fails:
        wrapper.write_command("tar", "exit 2\n")
    wrapper.env["TEST_STATUS"] = str(status)
    wrapper.write_parset()
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == status, result.stderr
    assert list(wrapper.work.glob("tmp.*")) == []
    if status:
        assert f"Rapthor execution failed, exit code: {status}" in result.stderr
    if archive_fails:
        assert "tar-ball may be incomplete" in result.stderr
    else:
        wrapper.assert_archive()


@pytest.mark.parametrize(
    "args, message",
    [
        ([], "Usage:"),
        (["input.parset", "input.parset"], "Usage:"),
        (["missing.parset"], "not found"),
    ],
    ids=["no-arguments", "extra-argument", "missing-parset"],
)
def test_invalid_arguments(wrapper, args, message):
    wrapper.write_parset()
    result = wrapper.run_wrapper(*args)
    assert result.returncode == 1, result.stderr
    assert message in result.stderr
    assert not wrapper.arguments.exists()
    assert not wrapper.work.exists()


@pytest.mark.parametrize(
    "contents",
    [
        "# dir_working = ignored",
        "dir_working = ",
        "dir_working = relative/path",
        "dir_working = None",
        "[imaging]\ndir_working = {work}",
    ],
    ids=["commented", "empty", "relative", "none", "wrong-section"],
)
def test_invalid_working_directory(wrapper, contents):
    wrapper.parset.write_text("[global]\n" + contents.format(work=wrapper.work) + "\n")
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == 1, result.stderr
    assert "Working directory invalid or not set" in result.stderr
    assert not wrapper.arguments.exists()
    assert not wrapper.work.exists()


@pytest.mark.parametrize("failure", ["environment", "versions"])
def test_setup_failure(wrapper, failure):
    if failure == "environment":
        wrapper.env["RAPTHOR_RC"] = str(wrapper.root / "missing.rc")
    else:
        wrapper.write_command("python", "exit 23\n")
    wrapper.write_parset()
    result = wrapper.run_wrapper(wrapper.parset)
    assert result.returncode == (1 if failure == "environment" else 23), result.stderr
    assert not wrapper.arguments.exists()
    assert not wrapper.work.exists() or list(wrapper.work.iterdir()) == []
