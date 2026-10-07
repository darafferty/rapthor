"""Exercise Docker label extraction without requiring Docker."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ExtractVersionHashesTest(unittest.TestCase):
    def test_label_extraction(self):
        script = Path(__file__).resolve().parents[2] / "Docker/extract_version_hashes.sh"
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            docker.write_text(
                '#!/bin/bash\nprintf "%s" "$MOCK_LABELS"\n'
                'if [ "$MOCK_STATUS" -ne 0 ]; then\n'
                '  echo "Error: Docker inspection failed" >&2\n'
                'fi\nexit "$MOCK_STATUS"\n'
            )
            docker.chmod(0o755)
            cases = [
                (
                    "matching",
                    "nl.astron.rapthor.dp3.version=abc123\n"
                    "nl.astron.rapthor.wsclean-idg.version=release=1\n",
                    0,
                    "DP3_COMMIT=abc123\nWSCLEANIDG_COMMIT=release=1\n",
                ),
                ("no labels", "", 0, None),
                ("unrelated labels", "other.version=abc\n", 0, None),
                ("Docker failure", "", 2, None),
                (
                    "Docker failure with partial output",
                    "nl.astron.rapthor.dp3.version=abc123\n",
                    2,
                    None,
                ),
            ]
            for name, labels, status, output in cases:
                with self.subTest(name=name):
                    env = dict(
                        os.environ,
                        PATH=directory + os.pathsep + os.environ["PATH"],
                        MOCK_LABELS=labels,
                        MOCK_STATUS=str(status),
                    )
                    result = subprocess.run(
                        ["bash", str(script), "test:image"],
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if status != 0:
                        self.assertEqual(result.returncode, status)
                        self.assertEqual(result.stdout, "")
                        self.assertEqual(result.stderr, "Error: Docker inspection failed\n")
                    elif output is None:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(result.stdout, "")
                        self.assertIn("no Rapthor version labels found", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0)
                        self.assertEqual(result.stdout, output)
                        self.assertEqual(result.stderr, "")
