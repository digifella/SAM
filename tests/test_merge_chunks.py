from __future__ import annotations

import unittest

import numpy as np

from run_sam_interactive import merge_chunks


def _chunk_lengths(total_duration: float, sample_rate: int, chunk_duration: float, overlap: float) -> list:
    """Mirror chunk_audio_generator's start/end/frame-count arithmetic
    (run_sam_interactive.py:705-741) without needing a real audio file, so
    the synthetic chunks below are sized exactly as the real generator would
    produce them for a file of the given duration.
    """
    lengths = []
    start = 0.0
    step = chunk_duration - overlap
    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        num_frames = int((end - start) * sample_rate)
        lengths.append(num_frames)
        start += step
    return lengths


def _synthetic_chunks(lengths: list, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n).astype(np.float32) for n in lengths]


def _reference_pre_fix_merge_chunks(chunks: list, overlap: float, sample_rate: int) -> np.ndarray:
    """Byte-for-byte copy of merge_chunks as it existed before the F1 fix
    (run_sam_interactive.py @ 76937fe). Kept here ONLY as a regression oracle
    to prove the fix does not perturb output for the currently-working
    low-overlap default profile. If merge_chunks changes again for a real
    reason, do not "fix" this copy to match — replace it with a fresh copy
    of the prior implementation instead, so it keeps meaning "the old code".
    """
    if len(chunks) == 1:
        return chunks[0]

    overlap_samples = int(overlap * sample_rate)
    total_samples = sum(len(c) for c in chunks) - (len(chunks) - 1) * overlap_samples
    merged = np.zeros(total_samples, dtype=chunks[0].dtype)

    pos = 0
    for i, chunk in enumerate(chunks):
        if i == 0:
            merged[:len(chunk)] = chunk
            pos = len(chunk) - overlap_samples
        else:
            actual_overlap = min(overlap_samples, len(chunk))
            if actual_overlap > 0:
                fade_out = np.linspace(1, 0, actual_overlap)
                fade_in = np.linspace(0, 1, actual_overlap)
                merged[pos:pos + actual_overlap] = (
                    merged[pos:pos + actual_overlap] * fade_out +
                    chunk[:actual_overlap] * fade_in
                )
                chunk_remainder = chunk[actual_overlap:]
                end_pos = pos + len(chunk)
                if end_pos > len(merged):
                    extra_samples = end_pos - len(merged)
                    merged = np.concatenate([merged, np.zeros(extra_samples, dtype=merged.dtype)])
                merged[pos + actual_overlap:end_pos] = chunk_remainder
            else:
                end_pos = pos + len(chunk)
                if end_pos > len(merged):
                    extra_samples = end_pos - len(merged)
                    merged = np.concatenate([merged, np.zeros(extra_samples, dtype=merged.dtype)])
                merged[pos:end_pos] = chunk
            pos += len(chunk) - actual_overlap

    return merged


class MergeChunksBufferSizingTests(unittest.TestCase):
    SAMPLE_RATE = 16000

    def test_high_overlap_10_9_no_longer_raises(self):
        # chunk_duration=10, overlap=9 on ~30.37s @ 16kHz: controller-verified
        # to currently compute a negative buffer length
        # (np.zeros(-180800, ...) -> ValueError: negative dimensions).
        lengths = _chunk_lengths(30.37, self.SAMPLE_RATE, 10, 9)
        self.assertEqual(len(lengths), 31)  # matches brief's verified chunk count
        chunks = _synthetic_chunks(lengths)

        merged = merge_chunks(chunks, overlap=9, sample_rate=self.SAMPLE_RATE)

        self.assertGreater(len(merged), 0)
        expected = int(30.37 * self.SAMPLE_RATE)
        self.assertLess(abs(len(merged) - expected), 200)  # small fp-accumulation drift only

    def test_high_overlap_30_25_no_longer_raises(self):
        # Milder ratio: doesn't go negative but currently under-allocates,
        # so the crossfade slice-assignment raises a broadcast ValueError.
        lengths = _chunk_lengths(30.37, self.SAMPLE_RATE, 30, 25)
        chunks = _synthetic_chunks(lengths)

        merged = merge_chunks(chunks, overlap=25, sample_rate=self.SAMPLE_RATE)

        self.assertGreater(len(merged), 0)
        expected = int(30.37 * self.SAMPLE_RATE)
        self.assertLess(abs(len(merged) - expected), 200)

    def test_default_profile_output_unchanged(self):
        # chunk_duration=60, overlap=2 (the currently-working default) must
        # produce byte-identical output to the pre-fix implementation.
        lengths = _chunk_lengths(185.3, self.SAMPLE_RATE, 60, 2)
        self.assertGreater(len(lengths), 1)  # exercise the real merge loop, not the early return
        chunks = _synthetic_chunks(lengths, seed=7)

        before = _reference_pre_fix_merge_chunks(chunks, overlap=2, sample_rate=self.SAMPLE_RATE)
        after = merge_chunks(chunks, overlap=2, sample_rate=self.SAMPLE_RATE)

        np.testing.assert_array_equal(before, after)

    def test_single_chunk_returns_input_unchanged(self):
        chunk = np.random.default_rng(1).standard_normal(1000).astype(np.float32)
        merged = merge_chunks([chunk], overlap=2, sample_rate=self.SAMPLE_RATE)
        self.assertIs(merged, chunk)


if __name__ == "__main__":
    unittest.main()
