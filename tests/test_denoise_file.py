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

    def test_trailing_partial_frame_silence_is_zeroed(self):
        """frame_rms_dbfs (and therefore silence_mask) only covers whole
        0.25s frames and drops any remainder, so a fixture whose total
        length lands exactly on a frame boundary can never exercise the
        trailing-partial-frame path. This fixture deliberately doesn't:
        it is the standard _make() fixture (17s, an exact multiple of the
        0.25s frame) plus a 5000-sample true-zero tail shorter than one
        frame -- the same shape that showed a ~-30 dBFS leak instead of
        exact zero before the tail was handled.
        """
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rng = np.random.default_rng(0)
            t = np.arange(int(5 * SR)) / SR
            speech = 0.2 * np.sin(2 * np.pi * 600 * t) + 0.01 * rng.standard_normal(len(t))
            noise = 0.01 * rng.standard_normal(int(4 * SR))
            silence = np.zeros(int(3 * SR))
            tail_len = 5000
            tail = np.zeros(tail_len)
            x = np.concatenate([speech, silence, noise, speech, tail]).astype(np.float32)
            n = max(1, int(0.25 * SR))
            self.assertNotEqual(len(x) % n, 0,
                                "fixture must NOT land on a frame boundary")

            src = td / "in.wav"
            sf.write(src, x, SR)
            d.denoise_file(src, td / "t.wav", td / "r.wav")
            out, _ = sf.read(td / "t.wav")
            self.assertEqual(len(out), len(x))
            tail_out = out[-tail_len:]
            self.assertLess(float(np.abs(tail_out).max()), 1e-4,
                            "trailing partial-frame silence must be zeroed too")

    def test_denoise_actually_reduces_noise_floor(self):
        """A no-op implementation (copy input to target, write a zero
        residual, fabricate the metadata dict) would pass every test above:
        the silence check holds trivially since that span is already silent
        in the input, and target+residual reconstructs the input trivially
        for target=input, residual=0. This test measures actual dB values
        in the noisy, non-silent, non-speech region that a no-op cannot
        satisfy: the noise floor must drop in the target, and the removed
        energy must show up in the residual.
        """
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._make(td)
            d.denoise_file(src, td / "t.wav", td / "r.wav")
            x = sf.read(src)[0].astype(np.float64)
            t = sf.read(td / "t.wav")[0].astype(np.float64)
            r = sf.read(td / "r.wav")[0].astype(np.float64)

            # Inside the noise region (8s-12s in _make), away from its edges
            # to avoid STFT boundary effects.
            a, b = int(8.5 * SR), int(11.5 * SR)

            def dbfs(seg):
                return 20.0 * np.log10(max(float(np.sqrt(np.mean(seg ** 2))), 1e-12))

            in_db = dbfs(x[a:b])
            target_db = dbfs(t[a:b])
            residual_db = dbfs(r[a:b])

            self.assertLess(
                target_db, in_db - 3.0,
                "target must be measurably quieter than the input in the "
                "noisy region -- a no-op passthrough is not enough")
            self.assertGreater(
                residual_db, in_db - 3.0,
                "residual must carry real energy removed from the noise, "
                "not be near-zero")


if __name__ == "__main__":
    unittest.main()
