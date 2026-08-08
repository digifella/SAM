"""On a removal job the guard must look at BOTH stems.

The derived prompt is "barking dog", which contains no speech word -- so the
description-driven arming in the original guard would never fire. That is the
exact hole that let the founding failure ship silently.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from worker.handlers import sam_audio_cleanup as h

SR = 16000
DUR = 4.0


def _tone(hz: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * DUR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float64)


def _write(tmp: Path, name: str, data: np.ndarray) -> Path:
    p = tmp / name
    sf.write(p, data, SR, subtype="PCM_16")
    return p


class RemovalGuardTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_speech_check_arms_without_a_speech_word(self):
        """A removal prompt never says "voice", so arming must not depend on it."""
        deliverable = _write(self.tmp, "t.wav", _tone(120))    # sub-300Hz rumble
        removed = _write(self.tmp, "r.wav", _tone(1000))       # speech band
        msg = h._separation_sanity_warning(
            deliverable, removed, "barking dog", remove_mode=True)
        self.assertIsNotNone(msg)

    def test_voice_in_the_removed_stem_is_reported(self):
        """The founding failure inverted: SAM decided the voice WAS the dog."""
        deliverable = _write(self.tmp, "t.wav", _tone(120))
        removed = _write(self.tmp, "r.wav", _tone(1000))
        msg = h._separation_sanity_warning(
            deliverable, removed, "barking dog", remove_mode=True)
        # Assert on wording UNIQUE to the mirror check. "removed" alone also
        # appears in the non-removal fallback message, so this test passed
        # against a mutant with the mirror branch deleted.
        self.assertIn("carries more speech than what was kept", msg)
        # The old wording points at the wrong file on a removal job.
        self.assertNotIn("voice is probably THERE", msg)

    def test_speech_free_removal_job_does_not_warn(self):
        """"Remove the drums from this instrumental" is not a failure.

        Arming the speech check unconditionally used to fire a third warning
        whenever the KEPT audio had no speech in it -- which, on material with
        no speech anywhere, is every single time. That message could only ever
        be reached when neither stem cleared the threshold, so it had no
        true-positive case; the mirror comparison above is self-calibrating and
        correctly declines here.
        """
        deliverable = _write(self.tmp, "t.wav", _tone(120))   # bass, no speech
        removed = _write(self.tmp, "r.wav", _tone(150))       # drums, no speech
        self.assertIsNone(h._separation_sanity_warning(
            deliverable, removed, "drums", remove_mode=True))

    def test_healthy_removal_is_silent(self):
        """Voice kept, tone removed -- nothing to warn about."""
        deliverable = _write(self.tmp, "t.wav", _tone(1000))
        removed = _write(self.tmp, "r.wav", _tone(120))
        msg = h._separation_sanity_warning(
            deliverable, removed, "barking dog", remove_mode=True)
        self.assertIsNone(msg)

    def test_near_silent_deliverable_warns_that_everything_was_removed(self):
        silent = _write(self.tmp, "t.wav", np.zeros(int(SR * DUR)))
        removed = _write(self.tmp, "r.wav", _tone(1000))
        msg = h._separation_sanity_warning(
            silent, removed, "barking dog", remove_mode=True)
        self.assertIsNotNone(msg)
        # Wording unique to the removal variant. "near-silent" alone also
        # appears in the pre-existing non-removal message, so this test passed
        # against a mutant with the removal branch deleted.
        self.assertIn("Almost nothing is left after removing", msg)

    def test_normal_separate_job_is_unchanged(self):
        """remove_mode defaults False: existing behaviour must not shift."""
        rumble = _write(self.tmp, "t.wav", _tone(120))
        voice = _write(self.tmp, "r.wav", _tone(1000))
        # No speech word -> no warning, exactly as before.
        self.assertIsNone(
            h._separation_sanity_warning(rumble, voice, "a dog barking"))
        # Speech word -> the original warning, pointing at residual.wav.
        msg = h._separation_sanity_warning(rumble, voice, "a person speaking")
        self.assertIsNotNone(msg)
        self.assertIn("voice is probably THERE", msg)


if __name__ == "__main__":
    unittest.main()
