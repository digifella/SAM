"""F5: _is_cap_limited_cuda_oom must correctly distinguish a CUDA OOM caused
by our own artificially low per-process memory_fraction cap from genuine GPU
exhaustion, against the REAL torch 2.6.0+cu124 CUDA OOM message format.

The message below was captured empirically (not guessed from a fragment
list) by setting a tiny torch.cuda.set_per_process_memory_fraction cap and
triggering a real oversized allocation on the pinned torch build. It was
captured twice, once while the card was under contention from an unrelated
process (~13 GB reported free by nvidia-smi) and once on a clean card
(~40 GB free); both captures were byte-for-byte identical, confirming the
"total capacity" / "is free" figures torch reports come from a source that
does not vary with the exact free-memory reading at capture time -- so this
test message is representative regardless of card contention.
"""

from __future__ import annotations

import unittest

import worker.handlers.sam_audio_cleanup as h

# Verbatim, captured via:
#   torch.cuda.set_per_process_memory_fraction(0.05, 0)
#   x = torch.empty(int(20e9 // 2), dtype=torch.float16, device="cuda")
REAL_CAP_LIMITED_OOM_MESSAGE = (
    "CUDA out of memory. Tried to allocate 18.63 GiB. GPU 0 has a total "
    "capacity of 45.00 GiB of which 43.66 GiB is free. Process 1459 has "
    "17179869184.00 GiB memory in use. 2.25 GiB allowed; Of the allocated "
    "memory 0 bytes is allocated by PyTorch, and 0 bytes is reserved by "
    "PyTorch but unallocated. If reserved but unallocated memory is large "
    "try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid "
    "fragmentation.  See documentation for Memory Management  "
    "(https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)"
)


class CapLimitedCudaOomDetectionTests(unittest.TestCase):
    def test_real_captured_cap_limited_message_returns_true(self):
        # (a) the real captured message: device reports 43.66 GiB free,
        # comfortably more than the 18.63 GiB requested -- this is our own
        # cap biting, not genuine exhaustion.
        exc = RuntimeError(REAL_CAP_LIMITED_OOM_MESSAGE)
        self.assertTrue(h._is_cap_limited_cuda_oom(exc))

    def test_genuine_exhaustion_style_message_returns_false(self):
        # (b) same fragment structure, but "is free" reports an amount
        # smaller than "Tried to allocate" -- the device is genuinely full,
        # so this must NOT be classified as cap-limited.
        text = (
            "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a "
            "total capacity of 48.00 GiB of which 0.50 GiB is free. "
            "Process 999 has 47.00 GiB memory in use. Of the allocated "
            "memory 46.80 GiB is allocated by PyTorch, and 0.20 GiB is "
            "reserved by PyTorch but unallocated."
        )
        exc = RuntimeError(text)
        self.assertFalse(h._is_cap_limited_cuda_oom(exc))

    def test_unrelated_error_returns_false_without_raising(self):
        # (c) a garbage/unrelated RuntimeError must degrade to False, never
        # raise -- an unparseable message must fall back to the existing
        # chunk-shrinking retry behaviour.
        exc = RuntimeError("ffmpeg transcode failed: no such file or directory")
        self.assertFalse(h._is_cap_limited_cuda_oom(exc))

    def test_unparseable_cuda_oom_message_returns_false_without_raising(self):
        # A message that trips the cuda+out-of-memory keyword gate but has
        # no size figures to parse must also degrade safely to False.
        exc = RuntimeError("CUDA out of memory. driver reported a failure.")
        self.assertFalse(h._is_cap_limited_cuda_oom(exc))

    def test_gib_unit_allocation_message_parses_and_is_detected(self):
        # (d) a distinct GiB-unit cap-limited example (separate from the
        # real captured string above) to exercise the GiB parsing path on
        # its own -- the original regex only ever matched MiB allocations
        # and would silently fail to match this.
        text = (
            "CUDA out of memory. Tried to allocate 5.50 GiB. GPU 0 has a "
            "total capacity of 45.00 GiB of which 20.00 GiB is free. "
            "Process 42 has 4.00 GiB memory in use. 6.00 GiB allowed; Of "
            "the allocated memory 3.90 GiB is allocated by PyTorch, and "
            "0.10 GiB is reserved by PyTorch but unallocated."
        )
        exc = RuntimeError(text)
        self.assertTrue(h._is_cap_limited_cuda_oom(exc))

    def test_mib_unit_allocation_message_still_parses(self):
        # Small allocations format in MiB rather than GiB; both units must
        # be handled since torch switches units based on magnitude.
        text = (
            "CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a "
            "total capacity of 45.00 GiB of which 20.00 GiB is free. "
            "Process 42 has 4.00 GiB memory in use."
        )
        exc = RuntimeError(text)
        self.assertTrue(h._is_cap_limited_cuda_oom(exc))


if __name__ == "__main__":
    unittest.main()
