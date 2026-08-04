"""Measured against the real failure case and Paul's WavePad result.

Fionnuala_raw.wav is 9m39s of one speaker with broadband noise, assembled from
separate cuts so 25% of it is absolute digital zero. SAM-Audio returned a target
with 0.1% speech-band energy on this file. Paul's WavePad spectral subtraction
correlates 0.978 with the raw input -- it removed noise without touching the voice.
That is the bar.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import sam_audio_utils.denoise as d
from worker.handlers.sam_audio_cleanup import _speech_band_fraction

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "Fionnuala_raw.wav"
ORACLE = REPO / "Fionnuala_raw_spectral subtraction.wav"


def _mono(p):
    x, sr = sf.read(p)
    return (x.mean(axis=1) if x.ndim > 1 else x).astype(np.float64), sr


@unittest.skipUnless(RAW.exists() and ORACLE.exists(), "real fixtures not present")
class RealAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        td = Path(cls.td.name)
        cls.target = td / "t.wav"
        cls.residual = td / "r.wav"
        cls.meta = d.denoise_file(RAW, cls.target, cls.residual)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_noise_window_is_not_digital_silence(self):
        self.assertGreater(self.meta["noise_window_dbfs"], -90.0,
                           "profile came from a zero run -- the gate failed")
        self.assertLess(self.meta["noise_window_dbfs"], -25.0,
                        "profile window is too loud to be noise")

    def test_speech_is_retained(self):
        frac = _speech_band_fraction(self.target)
        self.assertGreater(frac, 0.30,
                           f"speech-band energy collapsed to {frac:.1%} -- the voice "
                           f"was removed (SAM's failure was 0.1%)")

    def test_correlates_with_the_wavepad_oracle(self):
        ref, _ = _mono(ORACLE)
        out, _ = _mono(self.target)
        n = min(len(ref), len(out))
        a, b = ref[:n], out[:n]
        a = (a - a.mean()) / (a.std() or 1)
        b = (b - b.mean()) / (b.std() or 1)
        c = float(np.mean(a * b))
        self.assertGreater(c, 0.70,
                           f"correlation with the known-good result is only {c:.3f}")

    def test_noise_floor_is_reduced(self):
        raw, sr = _mono(RAW)
        out, _ = _mono(self.target)
        db_raw = d.frame_rms_dbfs(raw, sr)
        db_out = d.frame_rms_dbfs(out, sr)
        live = db_raw > d.SILENCE_GATE_DBFS
        floor_raw = np.percentile(db_raw[live], 10)
        floor_out = np.percentile(db_out[live], 10)
        self.assertLess(floor_out, floor_raw - 3.0,
                        f"noise floor barely moved: {floor_raw:.1f} -> {floor_out:.1f} dBFS")

    def test_silence_regions_stay_silent(self):
        raw, sr = _mono(RAW)
        out, _ = _mono(self.target)
        sil = d.silence_mask(raw, sr)
        n = int(0.25 * sr)
        idx = np.where(sil)[0]
        self.assertTrue(len(idx) > 0)
        for i in idx[:20]:
            seg = out[i * n:(i + 1) * n]
            self.assertLess(float(np.abs(seg).max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
