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


def _band_mix(speech_fraction: float, amp: float = 0.5) -> np.ndarray:
    """A stem with a chosen share of its energy inside the speech band.

    1kHz stands in for the voice, 6kHz for the out-of-band source (cicadas,
    cymbals). Amplitudes are sqrt-weighted because the fraction is a POWER
    ratio, which is what _speech_band_fraction measures.
    """
    return (_tone(1000.0, amp * float(np.sqrt(speech_fraction)))
            + _tone(6000.0, amp * float(np.sqrt(1.0 - speech_fraction))))


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

        Arming the speech check unconditionally fires the kept-stem warning
        whenever the KEPT audio has no speech in it -- which, on material with
        no speech anywhere, is every single time. The backstop is therefore
        gated on ABSOLUTE speech-band level, and here there is none to find:
        both stems measure about -104 dBFS in the band, 44 dB below the
        near-silence floor.
        """
        deliverable = _write(self.tmp, "t.wav", _tone(120))   # bass, no speech
        removed = _write(self.tmp, "r.wav", _tone(150))       # drums, no speech
        self.assertIsNone(h._separation_sanity_warning(
            deliverable, removed, "drums", remove_mode=True))

    def test_voice_swamped_in_the_removed_stem_is_still_reported(self):
        """The mirror check's blind spot: a share cannot see a swamped voice.

        _speech_band_fraction is a RATIO of the stem's own energy, so a stem can
        hold the entire voice and still score under MIN_SPEECH_BAND_FRACTION
        once a louder out-of-band source sits on top of it -- which is the
        normal state of the stem a removal job discards. Measured here:

            kept   : -13.4 dBFS RMS, 0.4% speech band (loud rumble, trace voice)
            removed:  -8.9 dBFS RMS, 3.9% speech band (the VOICE + louder 6kHz)

        3.9% is below the 10% arming threshold, so the mirror check declines and
        the voice would leave silently. The absolute measure sees it: the
        discarded stem holds 14 dB more speech-band energy than the kept one.
        """
        deliverable = _write(self.tmp, "t.wav",
                             _tone(80, 0.3005) + _tone(1000, 0.02))
        removed = _write(self.tmp, "r.wav",
                         _tone(1000, 0.1) + _tone(6000, 0.5))
        # Pin the premise: the mirror check genuinely cannot fire on this pair.
        self.assertLess(h._speech_band_fraction(removed), h.MIN_SPEECH_BAND_FRACTION)
        msg = h._separation_sanity_warning(
            deliverable, removed, "cicadas", remove_mode=True)
        self.assertIsNotNone(msg)
        self.assertIn("more speech-band energy", msg)

    def test_real_world_good_removal_is_within_the_mirror_margin(self):
        """Measured 2026-08-08 on Original_Audio.wav, and judged good by ear.

        "Remove the cicadas" from a boy talking under very loud cicadas gave a
        removed stem at 53.7% speech band against a kept stem at 48.8%. The
        listener's verdict on the deliverable was "did a great job", so a
        warning here is a false positive -- and a guard that fires on good
        output is one nobody reads. 4.9 points of excess must not be enough.
        """
        deliverable = _write(self.tmp, "t.wav", _band_mix(0.488))
        removed = _write(self.tmp, "r.wav", _band_mix(0.537))
        self.assertAlmostEqual(h._speech_band_fraction(removed), 0.537, places=2)
        self.assertAlmostEqual(h._speech_band_fraction(deliverable), 0.488, places=2)
        self.assertIsNone(h._separation_sanity_warning(
            deliverable, removed, "cicadas", remove_mode=True))

    def test_founding_failure_gap_still_trips_the_mirror_check(self):
        """The other end of the margin: 58.5 points must still warn loudly.

        Measured 2026-08-04 on the failure this whole guard exists for -- the
        residual carried 58.6% speech-band energy against a target at 0.1%.
        A margin wide enough to silence the 4.9-point false positive must still
        leave this tripping with room to spare.
        """
        deliverable = _write(self.tmp, "t.wav",
                             _tone(120, 0.3) + _tone(1000, 0.0095))
        removed = _write(self.tmp, "r.wav", _band_mix(0.586))
        msg = h._separation_sanity_warning(
            deliverable, removed, "cicadas", remove_mode=True)
        self.assertIsNotNone(msg)
        self.assertIn("carries more speech than what was kept", msg)

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
