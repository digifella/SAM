from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from worker.handlers.sam_audio_cleanup import _parse_loudnorm_json

SAMPLE_STDERR = """\
ffmpeg version 6.1.1 Copyright (c) 2000-2023
  Stream #0:0: Audio: pcm_s16le, 48000 Hz, mono, s16
[Parsed_loudnorm_0 @ 0x5b7e8c]
{
    "input_i" : "-27.61",
    "input_tp" : "-4.47",
    "input_lra" : "6.30",
    "input_thresh" : "-38.21",
    "output_i" : "-16.58",
    "output_tp" : "-3.00",
    "output_lra" : "5.90",
    "output_thresh" : "-27.07",
    "normalization_type" : "dynamic",
    "target_offset" : "0.58"
}
"""


class ParseLoudnormJsonTests(unittest.TestCase):
    def test_parses_measured_values(self):
        m = _parse_loudnorm_json(SAMPLE_STDERR)
        self.assertEqual(m["input_i"], "-27.61")
        self.assertEqual(m["input_tp"], "-4.47")
        self.assertEqual(m["input_lra"], "6.30")
        self.assertEqual(m["input_thresh"], "-38.21")
        self.assertEqual(m["target_offset"], "0.58")

    def test_no_json_block_raises(self):
        with self.assertRaises(ValueError):
            _parse_loudnorm_json("ffmpeg version 6.1.1\nno json here\n")

    def test_malformed_json_raises(self):
        with self.assertRaises(ValueError):
            _parse_loudnorm_json('prefix\n{\n    "input_i" : "-27.61",\n')

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            _parse_loudnorm_json('{\n  "input_i" : "-27.61"\n}\n')


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
class LoudnormRoundTripTests(unittest.TestCase):
    def test_quiet_tone_lands_at_minus_16_lufs(self):
        from worker.handlers.sam_audio_cleanup import (
            _apply_loudness_normalize,
            _measure_loudness,
        )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tone.wav"
            sr = 48000
            t = np.linspace(0, 4.0, 4 * sr, endpoint=False)
            sf.write(path, (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)

            _apply_loudness_normalize(path, "ffmpeg")

            measured = _measure_loudness(path, "ffmpeg")
            self.assertAlmostEqual(float(measured["input_i"]), -16.0, delta=1.0)
            self.assertLessEqual(float(measured["input_tp"]), -2.9)
            info = sf.info(str(path))
            self.assertEqual(info.samplerate, sr)


def _make_test_mp4(path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
        "-shortest", "-c:v", "libx264", "-c:a", "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
class ExtractAudioTests(unittest.TestCase):
    def test_mp4_extraction(self):
        from run_sam_interactive import extract_audio_to_wav

        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / "clip.mp4"
            _make_test_mp4(mp4)
            wav = extract_audio_to_wav(mp4, out_dir=Path(td))
            self.assertEqual(wav.name, "clip.wav")
            info = sf.info(str(wav))
            self.assertEqual(info.samplerate, 48000)
            self.assertEqual(info.channels, 2)
            self.assertAlmostEqual(info.duration, 2.0, delta=0.2)

    def test_mkv_extraction(self):
        from run_sam_interactive import extract_audio_to_wav

        with tempfile.TemporaryDirectory() as td:
            mp4 = Path(td) / "clip.mp4"
            _make_test_mp4(mp4)
            mkv = Path(td) / "clip.mkv"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp4), "-c", "copy", str(mkv)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            wav = extract_audio_to_wav(mkv, out_dir=Path(td))
            self.assertEqual(wav.name, "clip.wav")
            self.assertAlmostEqual(sf.info(str(wav)).duration, 2.0, delta=0.2)

    def test_no_audio_stream_raises(self):
        from run_sam_interactive import extract_audio_to_wav

        with tempfile.TemporaryDirectory() as td:
            silent = Path(td) / "noaudio.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                 "-c:v", "libx264", str(silent)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            with self.assertRaises(RuntimeError):
                extract_audio_to_wav(silent, out_dir=Path(td))


class VideoExtensionRoutingTests(unittest.TestCase):
    def test_find_audio_files_includes_video(self):
        from run_sam_interactive import find_audio_files

        with tempfile.TemporaryDirectory() as td:
            for name in ("a.wav", "b.mp4", "c.mkv", "d.txt", "e_target.wav"):
                (Path(td) / name).touch()
            found = {p.name for p in find_audio_files(Path(td))}
            self.assertEqual(found, {"a.wav", "b.mp4", "c.mkv"})


if __name__ == "__main__":
    unittest.main()
