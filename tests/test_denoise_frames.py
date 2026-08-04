from __future__ import annotations

import unittest

import numpy as np

import sam_audio_utils.denoise as d

SR = 16000


class FrameAnalysisTests(unittest.TestCase):
    def test_absolute_zero_frames_are_silence(self):
        x = np.concatenate([np.zeros(SR), 0.1 * np.ones(SR)]).astype(np.float32)
        mask = d.silence_mask(x, SR, frame_seconds=0.25)
        self.assertTrue(mask[:4].all(), "first second is absolute zero -> silence")
        self.assertFalse(mask[4:8].any(), "second second has signal -> not silence")

    def test_quiet_but_nonzero_is_not_silence(self):
        # -40 dBFS noise is the thing we want to MEASURE, not gate away.
        rng = np.random.default_rng(0)
        x = (0.01 * rng.standard_normal(SR)).astype(np.float32)
        self.assertFalse(d.silence_mask(x, SR).any(),
                         "-40 dBFS noise must not be classified as digital silence")

    def test_frame_rms_dbfs_matches_known_amplitude(self):
        x = (0.1 * np.ones(SR)).astype(np.float32)  # RMS 0.1 -> -20 dBFS
        db = d.frame_rms_dbfs(x, SR)
        self.assertTrue(np.allclose(db, -20.0, atol=0.1))

    def test_zero_frames_do_not_produce_nan_or_inf_in_stats(self):
        x = np.zeros(SR, dtype=np.float32)
        db = d.frame_rms_dbfs(x, SR)
        self.assertTrue(np.isfinite(db).all(), "silent frames must clamp, not go -inf")


if __name__ == "__main__":
    unittest.main()
