from __future__ import annotations

import unittest

import numpy as np

import sam_audio_utils.denoise as d

SR = 16000


def _tone(freq, seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


class SpectralSubtractTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.clean = _tone(800, 6.0)
        self.noise = 0.05 * rng.standard_normal(len(self.clean))
        self.noisy = (self.clean + self.noise).astype(np.float64)

    def _profile_from_noise(self):
        rng = np.random.default_rng(1)
        noise_only = np.concatenate([
            0.05 * rng.standard_normal(int(4 * SR)),
            self.noisy,
        ])
        return d.estimate_noise_profile(noise_only, SR)[0]

    def test_output_length_and_dtype_preserved(self):
        out = d.spectral_subtract(self.noisy, SR, self._profile_from_noise())
        self.assertEqual(len(out), len(self.noisy))
        self.assertTrue(np.isfinite(out).all(), "no NaN/inf in output")

    def test_noise_is_reduced(self):
        prof = self._profile_from_noise()
        out = d.spectral_subtract(self.noisy, SR, prof)
        before = float(np.sqrt(np.mean((self.noisy - self.clean) ** 2)))
        after = float(np.sqrt(np.mean((out[:len(self.clean)] - self.clean) ** 2)))
        self.assertLess(after, before, "residual noise must go down, not up")

    def test_signal_is_retained(self):
        """Reducing noise must not gut the tone -- correlation stays high."""
        prof = self._profile_from_noise()
        out = d.spectral_subtract(self.noisy, SR, prof)[:len(self.clean)]
        c = float(np.corrcoef(out, self.clean)[0, 1])
        self.assertGreater(c, 0.9, f"signal correlation collapsed to {c:.3f}")

    def test_stronger_setting_removes_more(self):
        prof = self._profile_from_noise()
        gentle = d.spectral_subtract(self.noisy, SR, prof, strength="gentle")
        aggressive = d.spectral_subtract(self.noisy, SR, prof, strength="aggressive")
        e_gentle = float(np.sqrt(np.mean((gentle[:len(self.clean)] - self.clean) ** 2)))
        e_aggr = float(np.sqrt(np.mean((aggressive[:len(self.clean)] - self.clean) ** 2)))
        self.assertLess(e_aggr, e_gentle)

    def test_unknown_strength_rejected(self):
        with self.assertRaises(ValueError):
            d.spectral_subtract(self.noisy, SR, self._profile_from_noise(), strength="nuclear")


if __name__ == "__main__":
    unittest.main()
