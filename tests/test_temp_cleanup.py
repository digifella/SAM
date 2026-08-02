from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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


def _fake_proc(returncode: int, stderr: str = "ffmpeg failed") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)


class ExtractAudioToWavOrphanedDirTests(unittest.TestCase):
    """F8: extract_audio_to_wav must not orphan a self-created temp dir when
    the ffmpeg step fails, but must never delete a caller-supplied out_dir."""

    def _leaked_dirs(self) -> set:
        return set(Path(tempfile.gettempdir()).glob("sam_video_audio_*"))

    def test_self_created_dir_removed_on_failure(self):
        from run_sam_interactive import extract_audio_to_wav

        before = self._leaked_dirs()
        with mock.patch(
            "run_sam_interactive.subprocess.run",
            return_value=_fake_proc(1),
        ):
            with self.assertRaises(RuntimeError):
                extract_audio_to_wav(Path("nonexistent_input.mp4"))

        self.assertEqual(self._leaked_dirs(), before)

    def test_caller_supplied_dir_survives_failure(self):
        from run_sam_interactive import extract_audio_to_wav

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            with mock.patch(
                "run_sam_interactive.subprocess.run",
                return_value=_fake_proc(1),
            ):
                with self.assertRaises(RuntimeError):
                    extract_audio_to_wav(Path("nonexistent_input.mp4"), out_dir=out_dir)

            self.assertTrue(out_dir.exists())


class AutoInputGainOrphanedDirTests(unittest.TestCase):
    """F8: auto_input_gain must not orphan a self-created temp dir when the
    pre-gain ffmpeg step fails, but must never delete a caller-supplied
    out_dir."""

    def _leaked_dirs(self) -> set:
        return set(Path(tempfile.gettempdir()).glob("sam_pregain_*"))

    # volumedetect stderr reporting a quiet peak, so decide_pregain_db()
    # returns a non-zero boost and the function proceeds to the pre-gain
    # ffmpeg step (the one under test).
    _QUIET_VOLUMEDETECT_STDERR = "[Parsed_volumedetect_0] max_volume: -30.0 dB"

    def test_self_created_dir_removed_on_failure(self):
        from run_sam_interactive import auto_input_gain

        before = self._leaked_dirs()
        with mock.patch(
            "run_sam_interactive.subprocess.run",
            side_effect=[
                _fake_proc(0, self._QUIET_VOLUMEDETECT_STDERR),  # volumedetect
                _fake_proc(1),  # pre-gain apply fails
            ],
        ):
            with self.assertRaises(RuntimeError):
                auto_input_gain(Path("nonexistent_input.wav"))

        self.assertEqual(self._leaked_dirs(), before)

    def test_caller_supplied_dir_survives_failure(self):
        from run_sam_interactive import auto_input_gain

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "pregain"
            with mock.patch(
                "run_sam_interactive.subprocess.run",
                side_effect=[
                    _fake_proc(0, self._QUIET_VOLUMEDETECT_STDERR),
                    _fake_proc(1),
                ],
            ):
                with self.assertRaises(RuntimeError):
                    auto_input_gain(Path("nonexistent_input.wav"), out_dir=out_dir)

            self.assertTrue(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
