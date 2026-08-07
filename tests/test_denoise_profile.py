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


class ShorterWindowFallbackTests(unittest.TestCase):
    """Audio whose noise-only pauses are shorter than the full window.

    Conversational audio with sub-3s pauses is ordinary, and refusing it
    outright rejected material that denoises well. Shorter windows are a
    fallback only -- noisier estimates, so they are warned about, and anything
    that succeeds at the full window is untouched.
    """

    def _pauses(self, pause_s, noise_amp):
        rng = np.random.default_rng(0)
        t = np.arange(int(90 * SR)) / SR
        env = np.ones(len(t))
        for g in np.arange(4, 90, 8):  # a noise-only pause every 8s
            env[int(g * SR):int((g + pause_s) * SR)] = 0
        speech = 0.2 * np.sin(2 * np.pi * 300 * t)
        return (speech * env + noise_amp * rng.standard_normal(len(t))).astype(np.float32)

    def test_falls_back_to_a_shorter_window_and_says_so(self):
        _, info = d.estimate_noise_profile(self._pauses(1.0, 0.03), SR)
        self.assertLess(info["window_seconds"], d.NOISE_WINDOW_SECONDS)
        self.assertGreaterEqual(info["headroom_db"], d.MIN_NOISE_HEADROOM_DB)
        self.assertTrue(any("noisier estimate" in w for w in info["warnings"]),
                        f"a shortened window must be warned about: {info['warnings']}")

    def test_full_window_is_preferred_and_unwarned(self):
        """The fallback must not fire on audio the full window handles."""
        rng = np.random.default_rng(1)
        x = np.concatenate([_speech(4, rng), _noise(8, 0.01, rng),
                            _speech(4, rng)]).astype(np.float32)
        _, info = d.estimate_noise_profile(x, SR)
        self.assertEqual(info["window_seconds"], d.NOISE_WINDOW_SECONDS)
        self.assertEqual(info["warnings"], [])

    def test_speech_only_audio_is_still_refused_at_every_length(self):
        """The fallback must not become a way to profile the speaker herself."""
        rng = np.random.default_rng(2)
        x = _speech(12, rng).astype(np.float32)
        with self.assertRaises(d.NoiseProfileError):
            d.estimate_noise_profile(x, SR)


if __name__ == "__main__":
    unittest.main()
