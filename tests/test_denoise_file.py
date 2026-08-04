from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import sam_audio_utils.denoise as d

SR = 44100


class DenoiseFileTests(unittest.TestCase):
    def _make(self, td: Path) -> Path:
        rng = np.random.default_rng(0)
        t = np.arange(int(5 * SR)) / SR
        speech = 0.2 * np.sin(2 * np.pi * 600 * t) + 0.01 * rng.standard_normal(len(t))
        noise = 0.01 * rng.standard_normal(int(4 * SR))
        silence = np.zeros(int(3 * SR))
        x = np.concatenate([speech, silence, noise, speech]).astype(np.float32)
        p = td / "in.wav"
        sf.write(p, x, SR)
        return p

    def test_writes_both_stems_at_input_sample_rate(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._make(td)
            meta = d.denoise_file(src, td / "t.wav", td / "r.wav")
            for name in ("t.wav", "r.wav"):
                data, sr = sf.read(td / name)
                self.assertEqual(sr, SR, "sample rate must be preserved")
                self.assertEqual(len(data), len(sf.read(src)[0]))
            self.assertEqual(meta["method"], "spectral_subtraction")
            self.assertEqual(meta["sample_rate"], SR)

    def test_true_silence_passes_through_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._make(td)
            d.denoise_file(src, td / "t.wav", td / "r.wav")
            out, _ = sf.read(td / "t.wav")
            sil = out[int(5 * SR) + 2000:int(8 * SR) - 2000]  # inside the zero run
            self.assertLess(float(np.abs(sil).max()), 1e-4,
                            "digital silence must stay silent, not gain artifacts")

    def test_target_plus_residual_reconstructs_the_input(self):
        """residual is what was REMOVED -- the pair must add back up."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._make(td)
            d.denoise_file(src, td / "t.wav", td / "r.wav")
            x = sf.read(src)[0].astype(np.float64)
            t = sf.read(td / "t.wav")[0].astype(np.float64)
            r = sf.read(td / "r.wav")[0].astype(np.float64)
            self.assertLess(float(np.abs((t + r) - x).max()), 1e-3)

    def test_progress_callback_is_called(self):
        seen = []
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._make(td)
            d.denoise_file(src, td / "t.wav", td / "r.wav",
                           progress_cb=lambda p, m: seen.append(p))
        self.assertTrue(seen, "progress_cb must be called at least once")


if __name__ == "__main__":
    unittest.main()
