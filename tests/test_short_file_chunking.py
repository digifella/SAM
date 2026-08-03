from __future__ import annotations

import unittest

import run_sam_interactive as r


class DecideChunkingRoutingTests(unittest.TestCase):
    """Task 9b Defect A: the non-chunked path intermittently produced a
    silent target (measured ~2/3 runs silent on identical input), while the
    chunked path was reliable (0/4 silent). Short files (duration <=
    chunk_duration) used to take the unreliable non-chunked path
    unconditionally. _decide_chunking is the pure routing decision
    process_audio_file now delegates to -- no audio I/O, no GPU -- so the
    "does a short file chunk, and with what effective chunk_duration" logic
    is directly testable here.
    """

    def test_long_file_unaffected(self):
        # duration > chunk_duration: must be byte-identical to the
        # pre-Task-9b behaviour (the validated 45-min soak path, cd=60 on a
        # 2708s input, must not move).
        use_chunking, eff_cd, eff_ov, forced = r._decide_chunking(
            duration=2708.0, chunk_duration=60.0, overlap=2.0
        )
        self.assertTrue(use_chunking)
        self.assertEqual(eff_cd, 60.0)
        self.assertEqual(eff_ov, 2.0)
        self.assertFalse(forced)

    def test_apollo13_cd30_regression_unchanged(self):
        # duration(57.64) > chunk_duration(30): already the "long file"
        # branch before Task 9b, so the forced-short-file logic must not
        # engage and chunk boundaries must not move.
        use_chunking, eff_cd, eff_ov, forced = r._decide_chunking(
            duration=57.64, chunk_duration=30.0, overlap=2.0
        )
        self.assertTrue(use_chunking)
        self.assertEqual(eff_cd, 30.0)
        self.assertEqual(eff_ov, 2.0)
        self.assertFalse(forced)

    def test_15s_clip_forces_chunking_with_shrunk_effective_duration(self):
        # duration(15) <= chunk_duration(60, the handler default): must now
        # force chunking instead of taking the unreliable non-chunked path.
        use_chunking, eff_cd, eff_ov, forced = r._decide_chunking(
            duration=15.0, chunk_duration=60.0, overlap=2.0
        )
        self.assertTrue(use_chunking)
        self.assertTrue(forced)
        self.assertEqual(eff_cd, 7.5)  # duration / 2
        self.assertEqual(eff_ov, 2.0)  # user's overlap fits comfortably under eff_cd
        # The chunk floor from Defect B must have real headroom, not be
        # right at the fold boundary.
        self.assertGreater(eff_cd, r.MIN_CHUNK_SECONDS)
        self.assertLess(eff_ov, eff_cd)

    def test_apollo13_duration_forces_chunking_when_below_default_chunk_duration(self):
        # duration(57.64) <= chunk_duration(60, the handler default): must
        # force chunking too (this is the actual Apollo13.wav-with-defaults
        # scenario that used to hit Defect A).
        use_chunking, eff_cd, eff_ov, forced = r._decide_chunking(
            duration=57.64, chunk_duration=60.0, overlap=2.0
        )
        self.assertTrue(use_chunking)
        self.assertTrue(forced)
        self.assertAlmostEqual(eff_cd, 28.82)
        self.assertEqual(eff_ov, 2.0)

    def test_large_overlap_is_capped_below_effective_chunk_duration(self):
        # overlap must remain strictly less than the effective (shrunk)
        # chunk_duration even when the caller's overlap was sized for the
        # original, larger chunk_duration -- e.g. overlap=25 is valid for
        # chunk_duration=30 but not for a shrunk ~7.5s effective duration.
        use_chunking, eff_cd, eff_ov, forced = r._decide_chunking(
            duration=15.0, chunk_duration=30.0, overlap=25.0
        )
        self.assertTrue(use_chunking)
        self.assertTrue(forced)
        self.assertEqual(eff_cd, 7.5)
        self.assertLess(eff_ov, eff_cd)

    def test_genuinely_tiny_input_falls_back_to_non_chunked_without_crashing(self):
        # A 3s voice note / 1s blip: half the duration is below the
        # forced-chunking threshold, so chunking would not buy reliability
        # (it would just fold back down to one chunk). Falling back to the
        # non-chunked path is the documented, accepted behaviour here --
        # what matters is it must not raise.
        for tiny_duration in (3.0, 1.0):
            with self.subTest(duration=tiny_duration):
                use_chunking, eff_cd, eff_ov, forced = r._decide_chunking(
                    duration=tiny_duration, chunk_duration=60.0, overlap=2.0
                )
                self.assertFalse(use_chunking)
                self.assertFalse(forced)

    def test_force_chunk_threshold_boundary(self):
        # duration=6.0 -> candidate half-duration is exactly
        # FORCE_CHUNK_MIN_SECONDS: still forces (>=). duration=5.9 doesn't.
        use_chunking_at, _, _, forced_at = r._decide_chunking(
            duration=2 * r.FORCE_CHUNK_MIN_SECONDS, chunk_duration=60.0, overlap=0.5
        )
        self.assertTrue(use_chunking_at)
        self.assertTrue(forced_at)

        use_chunking_below, _, _, forced_below = r._decide_chunking(
            duration=2 * r.FORCE_CHUNK_MIN_SECONDS - 0.2, chunk_duration=60.0, overlap=0.5
        )
        self.assertFalse(use_chunking_below)
        self.assertFalse(forced_below)


if __name__ == "__main__":
    unittest.main()
