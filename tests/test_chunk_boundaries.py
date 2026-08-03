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
