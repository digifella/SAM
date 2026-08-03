from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import run_sam_interactive as r


def _write_wav(path: Path, frames: int, sample_rate: int) -> None:
    sf.write(path, np.zeros(frames, dtype=np.float32), sample_rate)


class DegenerateTrailingChunkTests(unittest.TestCase):
    """Task 9b Defect B: chunk_duration=5, overlap=2 (step=3) on a ~15s clip
    emits a degenerate final chunk once the real file's duration lands a
    hair above an exact multiple of the step -- float accumulation of
    `step` plus a non-round sample count, both normal for a real recording.
    That degenerate chunk crashes sam_audio's codec reflect-pad (input
    shorter than the pad width). Reproduced here with the exact frame
    count/sample rate of the real apollo_15s.wav fixture that triggered the
    original crash (44100 Hz, 661504 frames -> 15.0000907s, not 15.0s).
    """

    def test_cd5_overlap2_no_degenerate_trailing_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            audio_path = Path(td) / "apollo_15s_like.wav"
            _write_wav(audio_path, frames=661504, sample_rate=44100)

            chunks = list(r.chunk_audio_generator(audio_path, chunk_duration=5.0, overlap=2.0))

            self.assertGreaterEqual(len(chunks), 1)
            for start, end, data, sr in chunks:
                duration = end - start
                self.assertGreaterEqual(
                    duration, r.MIN_CHUNK_SECONDS,
                    f"chunk [{start:.6f}, {end:.6f}) is {duration:.6f}s, below "
                    f"the {r.MIN_CHUNK_SECONDS}s floor -- would crash the codec reflect-pad",
                )
                # The chunk's actual sample count must match its declared
                # duration (folding must not desync the read from the boundary).
                self.assertAlmostEqual(len(data), round(duration * sr), delta=2)

            # No audio dropped: the last chunk must reach the true end of file.
            info = sf.info(audio_path)
            self.assertAlmostEqual(chunks[-1][1], info.duration, places=3)

    def test_cd7_overlap2_no_degenerate_trailing_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            audio_path = Path(td) / "apollo_15s_like.wav"
            _write_wav(audio_path, frames=661504, sample_rate=44100)

            chunks = list(r.chunk_audio_generator(audio_path, chunk_duration=7.0, overlap=2.0))

            self.assertGreaterEqual(len(chunks), 1)
            for start, end, data, sr in chunks:
                self.assertGreaterEqual(end - start, r.MIN_CHUNK_SECONDS)

    def test_subsecond_remainder_is_folded_not_dropped(self):
        # Deterministic (non-FP-edge) case: chunk_duration=5, overlap=2.9 on
        # an exact 15.0s file steps by 2.1s, landing a clean 0.3s remainder
        # (14.7 -> 15.0) that must be folded into the preceding chunk rather
        # than emitted standalone or silently dropped.
        with tempfile.TemporaryDirectory() as td:
            audio_path = Path(td) / "clean_15s.wav"
            _write_wav(audio_path, frames=15 * 16000, sample_rate=16000)

            chunks = list(r.chunk_audio_generator(audio_path, chunk_duration=5.0, overlap=2.9))

            self.assertGreater(len(chunks), 1)
            for start, end, data, sr in chunks:
                self.assertGreaterEqual(end - start, r.MIN_CHUNK_SECONDS)
            self.assertAlmostEqual(chunks[-1][1], 15.0, places=3)

    def test_count_chunks_matches_generator_after_folding(self):
        # count_chunks sizes process_audio_file's progress reporting before
        # chunk_audio_generator is iterated; they must agree on the
        # post-fold chunk count or the "chunk i/N" progress display desyncs.
        with tempfile.TemporaryDirectory() as td:
            audio_path = Path(td) / "apollo_15s_like.wav"
            _write_wav(audio_path, frames=661504, sample_rate=44100)

            n_generator = len(list(r.chunk_audio_generator(audio_path, chunk_duration=5.0, overlap=2.0)))
            n_count = r.count_chunks(audio_path, chunk_duration=5.0, overlap=2.0)
            self.assertEqual(n_generator, n_count)


if __name__ == "__main__":
    unittest.main()


class FoldPreservesMergedLengthTests(unittest.TestCase):
    """The brief singled out this invariant for the fold and nothing guarded it.

    Folding extends the previous chunk's end to total_duration. If that ever
    stops matching what merge_chunks reconstructs, audio is silently
    truncated -- the exact failure the fold exists to prevent.
    """

    # (total_duration, chunk_duration, overlap) -- each folds at least once.
    FOLDING_PROFILES = [
        (15.0, 5.0, 2.9),
        (15.0000907, 5.0, 2.0),
        (15.0000907, 7.0, 2.0),
        (120.4, 60.0, 2.0),
        (6.5, 3.25, 1.625),
    ]

    def test_merged_length_matches_total_duration_across_folding_profiles(self):
        sample_rate = 16000
        for total, cd, ov in self.FOLDING_PROFILES:
            with self.subTest(total=total, chunk_duration=cd, overlap=ov):
                boundaries = r._chunk_boundaries(total, cd, ov)
                # Build a chunk per boundary at its real sample length.
                chunks = [
                    np.zeros(int(round((end - start) * sample_rate)), dtype=np.float32)
                    for start, end in boundaries
                ]
                merged = r.merge_chunks(chunks, ov, sample_rate)
                expected = int(round(total * sample_rate))
                # Allow 1 sample of rounding drift, no more.
                self.assertLessEqual(
                    abs(len(merged) - expected), 1,
                    f"merged {len(merged)} samples vs expected {expected} "
                    f"for total={total} cd={cd} ov={ov} ({len(boundaries)} chunks)",
                )

    def test_fold_never_loses_the_tail(self):
        for total, cd, ov in self.FOLDING_PROFILES:
            with self.subTest(total=total, chunk_duration=cd, overlap=ov):
                boundaries = r._chunk_boundaries(total, cd, ov)
                self.assertAlmostEqual(
                    boundaries[-1][1], total, places=6,
                    msg="last boundary must reach the true end of the audio",
                )


class SoakBoundaryOracleTests(unittest.TestCase):
    """Pin the validated 45-minute capacity profile.

    The soak (2708s @ chunk_duration=60, overlap=2 -> 47 chunks, 2.65x
    realtime) is a measured capacity result. Nothing else in the suite stops
    a future chunking change from silently moving those boundaries, which
    would invalidate it without any test going red.
    """

    @staticmethod
    def _pre_fix_boundaries(total_duration, chunk_duration, overlap):
        """Preserved copy of the ORIGINAL stepping loop, before Task 9b.

        Deliberately NOT calling the production code -- comparing the new
        implementation against itself would prove nothing.

        NOTE: there is no early `break` when `end >= total_duration`. An
        earlier draft of this oracle had one and it wrongly reported the
        57.64s/cd=30 profile as changed. The real pre-fix generator kept
        stepping while `start < total_duration`, which is why that profile
        genuinely yields THREE chunks -- confirmed against a real run log
        for this exact input ("Will process 3 chunks ... chunk 3/3
        (56.0s - 57.6s)"). The production code was right; the oracle was wrong.
        """
        step = chunk_duration - overlap
        out = []
        start = 0.0
        while start < total_duration:
            end = min(start + chunk_duration, total_duration)
            out.append((start, end))
            start += step
        return out

    def test_soak_profile_boundaries_are_unchanged(self):
        total, cd, ov = 2708.891, 60.0, 2.0
        expected = self._pre_fix_boundaries(total, cd, ov)
        actual = [tuple(b) for b in r._chunk_boundaries(total, cd, ov)]
        self.assertEqual(len(actual), 47, "soak chunk count must stay 47")
        self.assertEqual(actual, expected, "soak chunk boundaries must not move")

    def test_long_file_profiles_are_unchanged(self):
        for total, cd, ov in [(2708.891, 60.0, 2.0), (185.3, 60.0, 2.0), (57.635986, 30.0, 2.0)]:
            with self.subTest(total=total):
                self.assertEqual(
                    [tuple(b) for b in r._chunk_boundaries(total, cd, ov)],
                    self._pre_fix_boundaries(total, cd, ov),
                )


class ChunkingConstantsTests(unittest.TestCase):
    """Literal-valued assertions so a silent retune of either constant is visible.

    The other tests derive their inputs from these constants, so they pass
    unchanged whatever the values are. Both constants are GPU-validated
    numbers, not free parameters.
    """

    def test_min_chunk_seconds_is_the_validated_value(self):
        self.assertEqual(r.MIN_CHUNK_SECONDS, 0.5)

    def test_force_chunk_min_seconds_is_the_validated_value(self):
        self.assertEqual(r.FORCE_CHUNK_MIN_SECONDS, 3.0)

    def test_min_chunk_floor_clears_the_codec_pad_requirement(self):
        # The codec reflect-pad needs input longer than hop_length-1 = 1919
        # samples (~40ms @ 48kHz). The floor must clear that with real margin.
        codec_floor_seconds = 1919 / 48000.0
        self.assertGreater(r.MIN_CHUNK_SECONDS, codec_floor_seconds * 10)
