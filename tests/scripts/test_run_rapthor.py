"""Exercise the shell entry point without a Rapthor software installation."""

import os
import shlex
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

WRAPPER = Path(__file__).resolve().parents[2] / "sbin" / "run-rapthor.sh"


class TestRunRapthor(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="run rapthor ")
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
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
        # Exercise leading whitespace, tabs around '=', trailing whitespace, and spaces in paths.
        self.parset.write_text(f"  dir_working\t=\t{self.work}   \n" + extra)

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
        self.assertEqual(len(archives), 1)
        self.assertRegex(archives[0].name, r"^rapthor_logs_\d{8}-\d{6}\.tar\.gz$")
        self.assertEqual(list(self.work.glob("tmp.*")), [])
        with tarfile.open(archives[0]) as archive:
            files = {member.name: member for member in archive if not member.isdir()}
            self.assertEqual(set(files), set(expected))
            for name, content in expected.items():
                self.assertTrue(files[name].isfile(), name)
                self.assertEqual(archive.extractfile(files[name]).read(), content)

    def expected_archive(self):
        return {
            "./parset": self.parset.read_bytes(),
            "./logs/nested/run.log": b"pipeline log\n",
        }

    def test_archive_dereferences_inputs_and_preserves_existing_links(self):
        self.work.mkdir()
        inputs = {}
        for name in ("input_skymodel", "apparent_skymodel", "strategy"):
            path = self.root / f"{name} file"
            path.write_text(f"{name} contents\n")
            inputs[name] = path
            (self.work / path.name).symlink_to(self.root / "missing old input")
        self.write_parset("".join(f"  {name}\t= {path}\n" for name, path in inputs.items()))
        result = self.run_wrapper(self.parset)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.arguments.read_text().splitlines(), ["-v", str(self.parset)])
        self.assertFalse((self.work / self.parset.name).exists())
        for path in inputs.values():
            self.assertTrue((self.work / path.name).is_symlink())
            self.assertEqual((self.work / path.name).readlink(), self.root / "missing old input")
        self.assert_archive(
            self.expected_archive()
            | {f"./{name}": path.read_bytes() for name, path in inputs.items()}
        )

    def test_existing_input_in_working_directory_is_preserved(self):
        self.work.mkdir()
        strategy = self.work / "custom strategy.py"
        strategy.write_text("strategy contents\n")
        self.write_parset(f"strategy = {strategy}\n")
        result = self.run_wrapper(self.parset)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(strategy.is_symlink())
        self.assert_archive(self.expected_archive() | {"./strategy": strategy.read_bytes()})

    def test_optional_inputs_can_be_absent_none_or_builtin(self):
        for extra in (
            "",
            "input_skymodel = None\napparent_skymodel = None\nstrategy = default\n",
        ):
            with self.subTest(extra=extra):
                for archive in self.work.glob("rapthor_logs_*.tar.gz"):
                    archive.unlink()
                self.write_parset(extra)
                result = self.run_wrapper(self.parset)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assert_archive(self.expected_archive())

    def test_missing_optional_file_warns_and_still_archives(self):
        missing = self.root / "missing skymodel"
        self.write_parset(f"input_skymodel = {missing}\n")
        result = self.run_wrapper(self.parset)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("tar-ball may be incomplete", result.stderr)
        self.assertFalse((self.work / missing.name).is_symlink())
        self.assert_archive(self.expected_archive())

    def test_failure_status_is_preserved_after_archiving(self):
        self.write_parset()
        self.env["TEST_STATUS"] = "37"
        result = self.run_wrapper(self.parset)
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertIn("Rapthor execution failed, exit code: 37", result.stderr)
        self.assert_archive(self.expected_archive())

    def test_archive_without_logs(self):
        self.write_command("rapthor", "exit 0\n")
        self.write_parset()
        result = self.run_wrapper(self.parset)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("tar-ball may be incomplete", result.stderr)
        self.assert_archive({"./parset": self.parset.read_bytes()})

    def test_invalid_arguments_and_working_directory_fail_before_rapthor(self):
        for args, contents, message in (
            ([], "", "Usage:"),
            ([self.parset, self.parset], "", "Usage:"),
            ([self.root / "missing.parset"], "", "not found"),
            (
                [self.parset],
                "# dir_working = ignored\n",
                "Working directory invalid or not set",
            ),
            (
                [self.parset],
                "dir_working =    \n",
                "Working directory invalid or not set",
            ),
            (
                [self.parset],
                "dir_working = relative/path\n",
                "Working directory invalid or not set",
            ),
            (
                [self.parset],
                "dir_working = None\n",
                "Working directory invalid or not set",
            ),
        ):
            with self.subTest(contents=contents, args=args):
                self.parset.write_text(contents)
                result = self.run_wrapper(*args)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.arguments.exists())
                self.assertFalse(self.work.exists())

    def test_relative_parset_is_resolved(self):
        self.write_parset()
        result = self.run_wrapper(self.parset.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.arguments.read_text().splitlines(), ["-v", str(self.parset)])
        self.assert_archive(self.expected_archive())

    def test_archive_failure_preserves_rapthor_status_and_cleans_staging(self):
        self.write_command("tar", "exit 2\n")
        self.write_parset()
        for status in (0, 37):
            with self.subTest(status=status):
                self.env["TEST_STATUS"] = str(status)
                result = self.run_wrapper(self.parset)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertIn("tar-ball may be incomplete", result.stderr)
                self.assertEqual(list(self.work.glob("tmp.*")), [])

    def test_setup_failure_cleans_staging(self):
        self.write_command("python", "exit 23\n")
        self.write_parset()
        result = self.run_wrapper(self.parset)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertFalse(self.arguments.exists())
        self.assertEqual(list(self.work.iterdir()), [])

    def test_missing_environment_fails_before_rapthor(self):
        self.env["RAPTHOR_RC"] = str(self.root / "missing.rc")
        self.write_parset()
        result = self.run_wrapper(self.parset)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to source rapthor environment", result.stderr)
        self.assertFalse(self.arguments.exists())
        self.assertFalse(self.work.exists())

    def test_inputs_with_same_basename_are_archived_separately(self):
        inputs = {}
        for name in ("input_skymodel", "apparent_skymodel", "strategy"):
            directory = self.root / name
            directory.mkdir()
            path = directory / "input"
            path.write_text(f"{name} contents\n")
            inputs[name] = path
        self.write_parset("".join(f"{name} = {path}\n" for name, path in inputs.items()))
        result = self.run_wrapper(self.parset)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_archive(
            self.expected_archive()
            | {f"./{name}": path.read_bytes() for name, path in inputs.items()}
        )

    def test_quoted_absolute_paths(self):
        inputs = {}
        for name in ("input_skymodel", "apparent_skymodel", "strategy"):
            path = self.root / f"{name} file"
            path.write_text(f"{name} contents\n")
            inputs[name] = path
        for quote in ("'", '"'):
            with self.subTest(quote=quote):
                for archive in self.work.glob("rapthor_logs_*.tar.gz"):
                    archive.unlink()
                self.parset.write_text(
                    f"[global]\n  dir_working = {quote}{self.work}{quote}   \n"
                    + "".join(
                        f"  {name} = {quote}{path}{quote}   \n" for name, path in inputs.items()
                    )
                )
                result = self.run_wrapper(self.parset)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("WARNING:", result.stderr)
                self.assert_archive(
                    self.expected_archive()
                    | {f"./{name}": path.read_bytes() for name, path in inputs.items()}
                )
