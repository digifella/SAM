from __future__ import annotations

import unittest

import numpy as np

import sam_audio_utils.denoise as d

SR = 16000


def _noise(seconds, amp, rng):
    return amp * rng.standard_normal(int(seconds * SR))


def _speech(seconds, rng):
    t = np.arange(int(seconds * SR)) / SR
    sig = sum(np.sin(2 * np.pi * f * t) for f in (400, 900, 1800))
    return 0.2 * sig / np.abs(sig).max() + 0.01 * rng.standard_normal(len(t))


class NoiseProfileTests(unittest.TestCase):
    def test_profile_is_not_taken_from_digital_silence(self):
        """The whole point: a zero run must never become the noise profile."""
        rng = np.random.default_rng(0)
        x = np.concatenate([
            _speech(4, rng),
            np.zeros(int(5 * SR)),      # true silence -- quieter than the noise
            _noise(4, 0.01, rng),       # the real noise floor
            _speech(4, rng),
        ]).astype(np.float32)

        profile, info = d.estimate_noise_profile(x, SR)

        self.assertGreater(profile.sum(), 0.0, "profile must not be all zeros")
        start = info["window_start_s"]
        self.assertFalse(4.0 <= start < 9.0,
                         f"window at {start}s is inside the digital-silence run")

    def test_raises_when_only_silence_and_speech_exist(self):
        """No quiet-but-nonzero region -> no honest profile. Fail loudly."""
        rng = np.random.default_rng(1)
        x = np.concatenate([_speech(4, rng), np.zeros(int(4 * SR)),
                            _speech(4, rng)]).astype(np.float32)
        with self.assertRaises(d.NoiseProfileError):
            d.estimate_noise_profile(x, SR)

    def test_raises_when_quietest_window_is_speech(self):
        """Guard against subtracting the speaker from herself."""
        rng = np.random.default_rng(2)
        x = _speech(12, rng).astype(np.float32)  # speech throughout, no quiet region
        with self.assertRaises(d.NoiseProfileError):
            d.estimate_noise_profile(x, SR)

    def test_info_reports_headroom_and_drift(self):
        rng = np.random.default_rng(3)
        x = np.concatenate([_speech(4, rng), _noise(6, 0.01, rng),
                            _speech(4, rng)]).astype(np.float32)
        _, info = d.estimate_noise_profile(x, SR)
        self.assertGreaterEqual(info["headroom_db"], d.MIN_NOISE_HEADROOM_DB)
        self.assertIn("drift_db", info)
        self.assertIsInstance(info["warnings"], list)


if __name__ == "__main__":
    unittest.main()
