"""Exercise Docker label extraction without requiring Docker."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ExtractVersionHashesTest(unittest.TestCase):
    def test_label_extraction(self):
        script = Path(__file__).resolve().parents[2] / "Docker/extract_version_hashes.sh"
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            docker.write_text(
                '#!/bin/bash\nprintf "%s" "$MOCK_LABELS"\nexit "${MOCK_STATUS:-0}"\n'
            )
            docker.chmod(0o755)
            cases = [
                ("matching", "nl.astron.rapthor.dp3.version=abc123\n"
                 "nl.astron.rapthor.wsclean-idg.version=release=1\n", 0,
                 "DP3_COMMIT=abc123\nWSCLEANIDG_COMMIT=release=1\n"),
                ("no labels", "", 0, None),
                ("unrelated labels", "other.version=abc\n", 0, None),
                ("Docker failure", "", 2, None),
            ]
            for name, labels, status, output in cases:
                with self.subTest(name=name):
                    env = dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"],
                               MOCK_LABELS=labels, MOCK_STATUS=str(status))
                    result = subprocess.run(
                        ["bash", str(script), "test:image"], env=env,
                        capture_output=True, text=True, check=False,
                    )
                    if output is None:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(result.stdout, "")
                        self.assertIn("no Rapthor version labels found", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0)
                        self.assertEqual(result.stdout, output)
                        self.assertEqual(result.stderr, "")
