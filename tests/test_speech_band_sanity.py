"""The near-silent guard only checked RMS, so a loud target with no speech in it
passed as healthy.

Real failure this encodes (2026-08-04, Fionnuala_raw.wav, 9m39s of one speaker
with broadband noise): asked for "a person speaking", SAM returned a target at
-21.6 dBFS RMS -- comfortably above the -60 dBFS silence floor, so the existing
guard said nothing. Measured band split of that target:

    <300Hz 93.5%   300-3400Hz 0.1%   >3400Hz 6.4%

i.e. it was room rumble with essentially zero speech. Her voice was in the
RESIDUAL (58.6% speech band). The pipeline reported success and the loud-but-
empty output was posted to Discord.

RMS cannot catch this: energy is not speech. The guard needs to look at where
the energy sits.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import worker.handlers.sam_audio_cleanup as h

SR = 16000


def _tone(freq: float, seconds: float = 2.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _speechlike(seconds: float = 2.0, amp: float = 0.3) -> np.ndarray:
    """Energy spread across the 300-3400Hz band, like voiced speech."""
    t = np.arange(int(seconds * SR)) / SR
    sig = sum(np.sin(2 * np.pi * f * t) for f in (400, 700, 1200, 1900, 2800))
    sig = sig / np.abs(sig).max()
    return (amp * sig).astype(np.float32)


class SpeechBandSanityTests(unittest.TestCase):
    def _write(self, td: Path, name: str, data: np.ndarray) -> Path:
        p = td / name
        sf.write(p, data, SR)
        return p

    def test_loud_rumble_target_is_flagged(self):
        """The Fionnuala case: loud target, no speech in it, speech in residual."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # 120Hz rumble at a healthy RMS -- passes the -60 dBFS silence gate.
            target = self._write(td, "t.wav", _tone(120))
            residual = self._write(td, "r.wav", _speechlike())

            warning = h._separation_sanity_warning(target, residual, "a person speaking")

            self.assertIsNotNone(
                warning,
                "a target that is 100% sub-300Hz rumble must be flagged even though "
                "its RMS is far above the silence floor",
            )
            self.assertIn("speech", warning.lower())
            self.assertIn("residual", warning.lower())

    def test_genuine_speech_target_is_not_flagged(self):
        """Must not fire on a good separation, or the warning becomes noise."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target = self._write(td, "t.wav", _speechlike())
            residual = self._write(td, "r.wav", _tone(120))
            self.assertIsNone(
                h._separation_sanity_warning(target, residual, "a person speaking"),
                "a target carrying real speech-band energy must not be flagged",
            )

    def test_near_silent_target_still_flagged_by_rms(self):
        """The original RMS guard must survive: silence is still silence."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target = self._write(td, "t.wav", _speechlike(amp=1e-5))
            residual = self._write(td, "r.wav", _speechlike())
            warning = h._separation_sanity_warning(target, residual, "a person speaking")
            self.assertIsNotNone(warning)
            self.assertIn("near-silent", warning.lower())

    def test_non_speech_description_is_not_flagged_for_lacking_speech(self):
        """Asking for a non-speech sound and getting one is correct, not a defect.

        The speech-band check must only apply when the description actually asks
        for speech -- otherwise extracting 'a dog barking' or 'bass guitar' would
        warn every time.
        """
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            target = self._write(td, "t.wav", _tone(120))
            residual = self._write(td, "r.wav", _speechlike())
            self.assertIsNone(
                h._separation_sanity_warning(target, residual, "a deep bass rumble"),
                "a non-speech description must not be judged against the speech band",
            )


if __name__ == "__main__":
    unittest.main()
