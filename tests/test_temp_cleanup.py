from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class ConvertToMono16kLeakTests(unittest.TestCase):
    """F7: convert_to_mono_16k must not leak its temp .wav when ffmpeg fails."""

    def _leaked_temp_files(self) -> set:
        return set(Path(tempfile.gettempdir()).glob("sam_mono16k_*"))

    def test_called_process_error_cleans_up_temp_file(self):
        from run_sam_interactive import convert_to_mono_16k

        before = self._leaked_temp_files()
        with mock.patch(
            "run_sam_interactive.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"boom"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                convert_to_mono_16k(Path("nonexistent_input.wav"))

        self.assertEqual(self._leaked_temp_files(), before)

    def test_timeout_cleans_up_temp_file(self):
        from run_sam_interactive import convert_to_mono_16k

        before = self._leaked_temp_files()
        with mock.patch(
            "run_sam_interactive.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=300),
        ):
            with self.assertRaises(subprocess.TimeoutExpired):
                convert_to_mono_16k(Path("nonexistent_input.wav"))

        self.assertEqual(self._leaked_temp_files(), before)


if __name__ == "__main__":
    unittest.main()
