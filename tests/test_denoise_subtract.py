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

    def test_gain_smoothing_reduces_frame_to_frame_envelope_roughness(self):
        """The 3-frame gain moving average is the spec's anti-warble mechanism.
        Deleting it left every other denoise test green (mutation-tested), so it
        was constraining nothing. This measures a smoothness-sensitive statistic
        of the OUTPUT itself -- the frame-envelope's total variation in dB -- via
        the public API only, on a stationary noise-only signal (the regime where
        frame-to-frame gain jitter, i.e. warble, actually shows up).

        Mutation-proven: deleting denoise.py's smoothing block pushes this
        roughness from ~0.54 to ~0.70 for this exact fixture, well past the
        threshold below (see progress ledger / PR description for the captured
        failure output).
        """
        rng = np.random.default_rng(42)
        amp = 0.05
        noise_only = amp * rng.standard_normal(int(4 * SR))
        noisy_for_profile = self.clean + amp * rng.standard_normal(len(self.clean))
        profile = d.estimate_noise_profile(
            np.concatenate([noise_only, noisy_for_profile]), SR)[0]
        # Same amplitude as the profile source, different draw: a stationary
        # noise-only signal, which is exactly the regime the spec names --
        # "isolated single-frame spikes are exactly what warbles."
        test_noise = amp * rng.standard_normal(int(6 * SR))

        out = d.spectral_subtract(test_noise, SR, profile, strength="normal")

        hop = d.STFT_WINDOW - d.STFT_OVERLAP
        n = len(out) // hop
        frames = out[:n * hop].reshape(n, hop)
        env_db = 20.0 * np.log10(np.maximum(np.sqrt((frames ** 2).mean(axis=1)), 1e-9))
        roughness = float(np.mean(np.abs(np.diff(env_db))))

        self.assertLess(
            roughness, 0.62,
            f"frame-to-frame envelope roughness {roughness:.3f} dB is too high "
            "-- gain smoothing should suppress isolated single-frame gain spikes")

    def test_spectral_floor_leaves_a_noise_bed(self):
        """beta (the spectral floor) is the spec's central anti-musical-noise
        mechanism: it stops bins being punched to zero, leaving a low noise bed
        instead. Only one existing test caught its absence, and only
        incidentally. This asserts the floor directly: on a stationary
        noise-only signal at the profile's own amplitude (so most bins compute
        a deeply negative raw gain and get clamped up to beta), the output must
        retain measurable energy rather than being driven toward silence.

        Mutation-proven: forcing beta=0 (removing the floor) drops this output
        from ~-59.7 dBFS to ~-73 dBFS for this exact fixture, well past the
        threshold below (see progress ledger / PR description for the captured
        failure output).
        """
        rng = np.random.default_rng(42)
        amp = 0.05
        noise_only = amp * rng.standard_normal(int(4 * SR))
        noisy_for_profile = self.clean + amp * rng.standard_normal(len(self.clean))
        profile = d.estimate_noise_profile(
            np.concatenate([noise_only, noisy_for_profile]), SR)[0]
        test_noise = amp * rng.standard_normal(int(6 * SR))

        # "aggressive" has the smallest beta (0.02), so it is the strength most
        # exposed by removing the floor entirely.
        out = d.spectral_subtract(test_noise, SR, profile, strength="aggressive")
        out_dbfs = 20.0 * np.log10(max(float(np.sqrt(np.mean(out ** 2))), 1e-12))

        self.assertGreater(
            out_dbfs, -65.0,
            f"output fell to {out_dbfs:.1f} dBFS -- the spectral floor should "
            "leave a noise bed rather than letting attenuated bins reach silence")


if __name__ == "__main__":
    unittest.main()
