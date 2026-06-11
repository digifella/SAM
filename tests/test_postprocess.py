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


VOLUMEDETECT_STDERR = """\
[Parsed_volumedetect_0 @ 0x60d1c4] n_samples: 2764800
[Parsed_volumedetect_0 @ 0x60d1c4] mean_volume: -38.2 dB
[Parsed_volumedetect_0 @ 0x60d1c4] max_volume: -28.5 dB
[Parsed_volumedetect_0 @ 0x60d1c4] histogram_28db: 21
"""


class ParseVolumedetectTests(unittest.TestCase):
    def test_parses_max_volume(self):
        from run_sam_interactive import parse_volumedetect_max_db

        self.assertEqual(parse_volumedetect_max_db(VOLUMEDETECT_STDERR), -28.5)

    def test_missing_max_volume_returns_none(self):
        from run_sam_interactive import parse_volumedetect_max_db

        self.assertIsNone(parse_volumedetect_max_db("no volume info here"))

    def test_inf_silence_returns_none(self):
        from run_sam_interactive import parse_volumedetect_max_db

        self.assertIsNone(parse_volumedetect_max_db("max_volume: -inf dB"))


class DecidePregainTests(unittest.TestCase):
    def test_quiet_input_boosted_to_target(self):
        from run_sam_interactive import decide_pregain_db

        self.assertAlmostEqual(decide_pregain_db(-28.5), 25.5)

    def test_boost_capped(self):
        from run_sam_interactive import decide_pregain_db

        self.assertAlmostEqual(decide_pregain_db(-50.0), 30.0)

    def test_healthy_level_untouched(self):
        from run_sam_interactive import decide_pregain_db

        self.assertEqual(decide_pregain_db(-4.4), 0.0)
        self.assertEqual(decide_pregain_db(-3.0), 0.0)
        self.assertEqual(decide_pregain_db(-6.0), 0.0)

    def test_hot_input_reduced_to_target(self):
        from run_sam_interactive import decide_pregain_db

        self.assertAlmostEqual(decide_pregain_db(0.0), -3.0)

    def test_unmeasurable_untouched(self):
        from run_sam_interactive import decide_pregain_db

        self.assertEqual(decide_pregain_db(None), 0.0)

    def test_silence_floor_untouched(self):
        from run_sam_interactive import decide_pregain_db

        self.assertEqual(decide_pregain_db(-91.0), 0.0)
        self.assertEqual(decide_pregain_db(-60.0), 0.0)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
class AutoInputGainTests(unittest.TestCase):
    def _write_tone(self, path: Path, amplitude: float) -> None:
        sr = 48000
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        sf.write(path, (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)

    def test_quiet_input_gets_boosted(self):
        from run_sam_interactive import auto_input_gain

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "quiet.wav"
            self._write_tone(src, 0.01)  # peak ~= -40 dBFS
            out, gain_db = auto_input_gain(src, out_dir=Path(td) / "gain")
            self.assertNotEqual(out, src)
            self.assertEqual(out.name, "quiet.wav")
            # peak -40 dBFS needs +37 dB to reach -3; boost is capped at +30
            self.assertAlmostEqual(gain_db, 30.0, delta=0.3)
            data, _ = sf.read(out)
            peak_db = 20 * np.log10(np.max(np.abs(data)))
            self.assertAlmostEqual(peak_db, -10.0, delta=1.5)  # -40 + 30 cap

    def test_healthy_input_passthrough(self):
        from run_sam_interactive import auto_input_gain

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "ok.wav"
            self._write_tone(src, 0.6)  # peak ~= -4.4 dBFS
            out, gain_db = auto_input_gain(src, out_dir=Path(td) / "gain")
            self.assertEqual(out, src)
            self.assertEqual(gain_db, 0.0)

    def test_silence_passthrough(self):
        from run_sam_interactive import auto_input_gain

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "silence.wav"
            sf.write(src, np.zeros(48000, dtype=np.float32), 48000)
            out, gain_db = auto_input_gain(src, out_dir=Path(td) / "gain")
            self.assertEqual(out, src)
            self.assertEqual(gain_db, 0.0)


if __name__ == "__main__":
    unittest.main()
