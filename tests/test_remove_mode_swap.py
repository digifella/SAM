"""remove_mode swaps which stem is the deliverable.

Everything in handle() after the stems resolve is asymmetric on target_path --
loudness normalization, sf.info metadata, and the ZIP member names all read it.
Swapping at the point of resolution is what makes those correct for free; this
test pins the orientation and the metadata contract.
"""
import inspect
import unittest

from worker.handlers import sam_audio_cleanup as h


class RemoveModeSignatureTests(unittest.TestCase):
    def test_handle_accepts_remove_mode_defaulting_false(self):
        sig = inspect.signature(h.handle)
        self.assertIn("remove_mode", sig.parameters)
        self.assertIs(sig.parameters["remove_mode"].default, False)

    def test_swap_happens_before_the_guard_and_loudnorm(self):
        """Order is the whole point: a later swap would normalize the wrong stem."""
        src = inspect.getsource(h.handle)
        swap = src.index("target_path, residual_path = residual_path, target_path")
        guard = src.index("_separation_sanity_warning(")
        loudnorm = src.index("_apply_loudness_normalize(")
        zip_write = src.index('arcname="target.wav"')
        self.assertLess(swap, guard)
        self.assertLess(swap, loudnorm)
        self.assertLess(swap, zip_write)


if __name__ == "__main__":
    unittest.main()
