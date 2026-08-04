# Spectral-Subtraction Denoiser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-only spectral-subtraction denoiser as the default cleanup path, because SAM-Audio is source separation and fails on single-speaker broadband noise.

**Architecture:** A standalone `sam_audio_utils/denoise.py` (numpy/scipy, no torch, no GPU) exposing pure functions for noise estimation and subtraction. `clean_cli.py` routes to it by default and to SAM only on positive evidence of a mixture. The existing output contract (`status.json`, `result.zip` with target/residual/metadata, `--opus`, webhook) is unchanged, so Cortex, the bridge, the sp4 wrapper and Hermes need no changes.

**Tech Stack:** Python 3.11, numpy, scipy 1.16.3 (`scipy.signal.stft`/`istft`), soundfile. Plain pytest/unittest, unittest-style classes.

**Spec:** `docs/superpowers/specs/2026-08-04-spectral-denoiser-design.md`

## Global Constraints

- Run everything with the repo venv: `.venv/bin/python`. Tests from the repo root: `.venv/bin/python -m pytest tests/ -v`.
- **Do NOT install or upgrade any package.** torch 2.6.0+cu124 and streamlit 1.19.0 are deliberately pinned; the fp16 build is validated on a Quadro RTX 8000.
- **`sam_audio_utils/denoise.py` must NOT import torch**, must not touch the GPU, and must not load any model. It is pure numpy/scipy/soundfile.
- **Sample rate is preserved end to end.** 44.1kHz in → 44.1kHz out. No resampling anywhere in the denoise path.
- Baseline before this plan: **101 passing tests, exactly 6 known third-party warnings** (pynvml, pkg_resources ×2, torch weight_norm, SWIG ×2). No regressions, no new warnings.
- Never `git add -A`. Stage by explicit path. `Apollo13.wav:Zone.Identifier`, `sam-audio-colab-upload.zip` and `spool/` must stay untracked.
- **Run every command in the FOREGROUND.** Do not background test runs. Long commands are expected; let them block.
- Commits end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Test fixtures: `Fionnuala_raw.wav` (9m39s, 44.1kHz, the real case), `Fionnuala_raw_spectral subtraction.wav` (Paul's WavePad oracle, note the space in the filename), `Apollo13.wav` (57.6s, a second, different input). All in the repo root.

## Named constants (define in `denoise.py`, use verbatim)

```python
SILENCE_GATE_DBFS = -90.0      # at/below this a frame is digital silence, not noise
NOISE_WINDOW_SECONDS = 3.0     # length of the noise-profile window
MIN_NOISE_HEADROOM_DB = 6.0    # profile must sit >= this far below the speech median
MAX_NOISE_DRIFT_DB = 10.0      # per-block quiet-floor spread above which we warn
STFT_WINDOW = 2048             # ~46ms at 44.1kHz
STFT_OVERLAP = 1536            # 75% overlap
STRENGTH_PRESETS = {           # (alpha over-subtraction, beta spectral floor)
    "gentle": (1.5, 0.10),
    "normal": (2.0, 0.05),
    "aggressive": (3.0, 0.02),
}
```

---

## Task 1: Frame analysis and the silence gate

The load-bearing piece. 25% of the real test file is absolute digital zero; if noise
statistics include those frames every estimator returns an empty profile and the
denoiser becomes a silent no-op that looks like success.

**Files:**
- Create: `sam_audio_utils/denoise.py`
- Test: `tests/test_denoise_frames.py`

**Interfaces:**
- Produces (relied on by Tasks 2, 3, 5):
  - `frame_rms_dbfs(x: np.ndarray, sr: int, frame_seconds: float = 0.25) -> np.ndarray`
  - `silence_mask(x: np.ndarray, sr: int, frame_seconds: float = 0.25) -> np.ndarray` — bool array, True where the frame is digital silence
  - Module constants above.

- [ ] **Step 1: Write the failing test**

Create `tests/test_denoise_frames.py`:

```python
from __future__ import annotations

import unittest

import numpy as np

import sam_audio_utils.denoise as d

SR = 16000


class FrameAnalysisTests(unittest.TestCase):
    def test_absolute_zero_frames_are_silence(self):
        x = np.concatenate([np.zeros(SR), 0.1 * np.ones(SR)]).astype(np.float32)
        mask = d.silence_mask(x, SR, frame_seconds=0.25)
        self.assertTrue(mask[:4].all(), "first second is absolute zero -> silence")
        self.assertFalse(mask[4:8].any(), "second second has signal -> not silence")

    def test_quiet_but_nonzero_is_not_silence(self):
        # -40 dBFS noise is the thing we want to MEASURE, not gate away.
        rng = np.random.default_rng(0)
        x = (0.01 * rng.standard_normal(SR)).astype(np.float32)
        self.assertFalse(d.silence_mask(x, SR).any(),
                         "-40 dBFS noise must not be classified as digital silence")

    def test_frame_rms_dbfs_matches_known_amplitude(self):
        x = (0.1 * np.ones(SR)).astype(np.float32)  # RMS 0.1 -> -20 dBFS
        db = d.frame_rms_dbfs(x, SR)
        self.assertTrue(np.allclose(db, -20.0, atol=0.1))

    def test_zero_frames_do_not_produce_nan_or_inf_in_stats(self):
        x = np.zeros(SR, dtype=np.float32)
        db = d.frame_rms_dbfs(x, SR)
        self.assertTrue(np.isfinite(db).all(), "silent frames must clamp, not go -inf")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_denoise_frames.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sam_audio_utils.denoise'`

- [ ] **Step 3: Write minimal implementation**

Create `sam_audio_utils/denoise.py`:

```python
"""Spectral-subtraction denoiser.

SAM-Audio is source separation and fails on single-speaker broadband noise.
This is the denoiser for that case. Pure numpy/scipy: no torch, no GPU, no model.
"""
from __future__ import annotations

import numpy as np

SILENCE_GATE_DBFS = -90.0
NOISE_WINDOW_SECONDS = 3.0
MIN_NOISE_HEADROOM_DB = 6.0
MAX_NOISE_DRIFT_DB = 10.0
STFT_WINDOW = 2048
STFT_OVERLAP = 1536
STRENGTH_PRESETS = {
    "gentle": (1.5, 0.10),
    "normal": (2.0, 0.05),
    "aggressive": (3.0, 0.02),
}

_FLOOR = 1e-12  # clamp so digital silence gives a finite dBFS, not -inf


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim > 1 else x


def frame_rms_dbfs(x: np.ndarray, sr: int, frame_seconds: float = 0.25) -> np.ndarray:
    """Per-frame RMS in dBFS. Silent frames clamp to a finite floor, never -inf."""
    x = _mono(x)
    n = max(1, int(frame_seconds * sr))
    count = len(x) // n
    if count == 0:
        return np.array([20.0 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), _FLOOR))])
    frames = x[:count * n].reshape(count, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, _FLOOR))


def silence_mask(x: np.ndarray, sr: int, frame_seconds: float = 0.25) -> np.ndarray:
    """True where a frame is digital silence rather than quiet noise.

    Excluding these is what stops the noise profile being estimated from zeros.
    """
    return frame_rms_dbfs(x, sr, frame_seconds) <= SILENCE_GATE_DBFS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_denoise_frames.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Verify against the REAL file — the numbers must match the spec**

Run:

```bash
.venv/bin/python -c "
import soundfile as sf, sam_audio_utils.denoise as d
x, sr = sf.read('Fionnuala_raw.wav')
m = d.silence_mask(x, sr)
print(f'silence frames: {m.sum()} / {len(m)} = {100*m.mean():.1f}%')
"
```
Expected: **25.0%** (579 of 2315). If it differs, the gate is wrong — stop and report.

- [ ] **Step 6: Commit**

```bash
git add sam_audio_utils/denoise.py tests/test_denoise_frames.py
git commit -m "feat: frame analysis and digital-silence gate for the denoiser"
```

---

## Task 2: Noise profile estimation with validation guards

**Files:**
- Modify: `sam_audio_utils/denoise.py`
- Test: `tests/test_denoise_profile.py`

**Interfaces:**
- Consumes: `frame_rms_dbfs`, `silence_mask` from Task 1.
- Produces (relied on by Tasks 3, 5):
  - `NoiseProfileError(Exception)` — raised when no usable profile exists
  - `estimate_noise_profile(x, sr) -> tuple[np.ndarray, dict]` — returns the mean magnitude spectrum over the chosen window (length `STFT_WINDOW // 2 + 1`) and an info dict with keys `window_start_s`, `window_dbfs`, `speech_median_dbfs`, `headroom_db`, `drift_db`, `warnings` (list of str).

- [ ] **Step 1: Write the failing test**

Create `tests/test_denoise_profile.py`:

```python
from __future__ import annotations

import unittest

import numpy as np

import sam_audio_utils.denoise as d

SR = 16000


def _noise(seconds, amp, rng):
    return amp * rng.standard_normal(int(seconds * SR))


def _speech(seconds, rng):
    t = np.arange(int(seconds * SR)) / SR
    sig = sum(np.sin(2 * np.pi * f * t) for f in (400, 900, 1800))
    return 0.2 * sig / np.abs(sig).max() + 0.01 * rng.standard_normal(len(t))


class NoiseProfileTests(unittest.TestCase):
    def test_profile_is_not_taken_from_digital_silence(self):
        """The whole point: a zero run must never become the noise profile."""
        rng = np.random.default_rng(0)
        x = np.concatenate([
            _speech(4, rng),
            np.zeros(int(5 * SR)),      # true silence -- quieter than the noise
            _noise(4, 0.01, rng),       # the real noise floor
            _speech(4, rng),
        ]).astype(np.float32)

        profile, info = d.estimate_noise_profile(x, SR)

        self.assertGreater(profile.sum(), 0.0, "profile must not be all zeros")
        start = info["window_start_s"]
        self.assertFalse(4.0 <= start < 9.0,
                         f"window at {start}s is inside the digital-silence run")

    def test_raises_when_only_silence_and_speech_exist(self):
        """No quiet-but-nonzero region -> no honest profile. Fail loudly."""
        rng = np.random.default_rng(1)
        x = np.concatenate([_speech(4, rng), np.zeros(int(4 * SR)),
                            _speech(4, rng)]).astype(np.float32)
        with self.assertRaises(d.NoiseProfileError):
            d.estimate_noise_profile(x, SR)

    def test_raises_when_quietest_window_is_speech(self):
        """Guard against subtracting the speaker from herself."""
        rng = np.random.default_rng(2)
        x = _speech(12, rng).astype(np.float32)  # speech throughout, no quiet region
        with self.assertRaises(d.NoiseProfileError):
            d.estimate_noise_profile(x, SR)

    def test_info_reports_headroom_and_drift(self):
        rng = np.random.default_rng(3)
        x = np.concatenate([_speech(4, rng), _noise(6, 0.01, rng),
                            _speech(4, rng)]).astype(np.float32)
        _, info = d.estimate_noise_profile(x, SR)
        self.assertGreaterEqual(info["headroom_db"], d.MIN_NOISE_HEADROOM_DB)
        self.assertIn("drift_db", info)
        self.assertIsInstance(info["warnings"], list)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_denoise_profile.py -v`
Expected: FAIL — `AttributeError: module 'sam_audio_utils.denoise' has no attribute 'estimate_noise_profile'`

- [ ] **Step 3: Write minimal implementation**

Append to `sam_audio_utils/denoise.py`:

```python
from scipy.signal import stft as _stft


class NoiseProfileError(Exception):
    """No honest noise profile could be estimated from this audio."""


def estimate_noise_profile(x: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """Average magnitude spectrum of the quietest NON-SILENT window.

    Digital-silence frames are excluded first: on a collation of cuts they are
    the quietest thing in the file, and a profile taken from them is all zeros,
    which subtracts nothing while appearing to succeed.
    """
    x = _mono(x)
    frame_s = 0.25
    db = frame_rms_dbfs(x, sr, frame_s)
    sil = db <= SILENCE_GATE_DBFS
    usable = ~sil
    if not usable.any():
        raise NoiseProfileError(
            "every frame is digital silence - there is no noise to profile")

    speech_median = float(np.median(db[usable]))

    win_frames = max(1, int(NOISE_WINDOW_SECONDS / frame_s))
    if usable.sum() < win_frames:
        raise NoiseProfileError(
            f"fewer than {NOISE_WINDOW_SECONDS}s of non-silent audio to profile")

    # Slide the window; only positions containing NO silent frame are eligible,
    # so the profile is never diluted by zeros.
    best_start, best_level = None, None
    for i in range(len(db) - win_frames + 1):
        seg_sil = sil[i:i + win_frames]
        if seg_sil.any():
            continue
        level = float(np.mean(db[i:i + win_frames]))
        if best_level is None or level < best_level:
            best_start, best_level = i, level
    if best_start is None:
        raise NoiseProfileError(
            f"no {NOISE_WINDOW_SECONDS}s window free of digital silence")

    headroom = speech_median - best_level
    if headroom < MIN_NOISE_HEADROOM_DB:
        raise NoiseProfileError(
            f"quietest window is only {headroom:.1f} dB below the speech median "
            f"(need {MIN_NOISE_HEADROOM_DB}); it likely contains speech, and "
            f"subtracting it would remove the speaker")

    a = int(best_start * frame_s * sr)
    b = int((best_start + win_frames) * frame_s * sr)
    _, _, Z = _stft(x[a:b], fs=sr, nperseg=STFT_WINDOW, noverlap=STFT_OVERLAP)
    profile = np.abs(Z).mean(axis=1)

    # Drift: spread of the quiet floor across 60s blocks, non-silent frames only.
    block = max(1, int(60.0 / frame_s))
    floors = []
    for s in range(0, len(db), block):
        seg = db[s:s + block][usable[s:s + block]]
        if len(seg):
            floors.append(float(np.percentile(seg, 10)))
    drift = (max(floors) - min(floors)) if len(floors) > 1 else 0.0

    warnings: list[str] = []
    if drift > MAX_NOISE_DRIFT_DB:
        warnings.append(
            f"noise floor drifts {drift:.1f} dB across the file (>{MAX_NOISE_DRIFT_DB}); "
            f"one global profile may not fit all of it")

    info = {
        "window_start_s": round(best_start * frame_s, 2),
        "window_dbfs": round(best_level, 1),
        "speech_median_dbfs": round(speech_median, 1),
        "headroom_db": round(headroom, 1),
        "drift_db": round(drift, 1),
        "warnings": warnings,
    }
    return profile, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_denoise_profile.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Verify against the REAL file**

Run:

```bash
.venv/bin/python -c "
import soundfile as sf, sam_audio_utils.denoise as d
x, sr = sf.read('Fionnuala_raw.wav')
p, info = d.estimate_noise_profile(x, sr)
print(info)
print('profile sum:', p.sum())
"
```
Expected: `profile sum` clearly > 0; `window_dbfs` around **−38** (NOT −240); `headroom_db` comfortably above 6; `drift_db` around 3. Report the actual numbers. If `window_dbfs` is near −240 the silence gate is not working — stop and report.

- [ ] **Step 6: Commit**

```bash
git add sam_audio_utils/denoise.py tests/test_denoise_profile.py
git commit -m "feat: noise profile estimation that excludes digital silence"
```

---

## Task 3: Spectral subtraction

**Files:**
- Modify: `sam_audio_utils/denoise.py`
- Test: `tests/test_denoise_subtract.py`

**Interfaces:**
- Consumes: `estimate_noise_profile`, `STRENGTH_PRESETS`, `STFT_WINDOW`, `STFT_OVERLAP`.
- Produces (relied on by Tasks 4, 5):
  - `spectral_subtract(x, sr, profile, strength="normal") -> np.ndarray` — same length and sample rate as `x`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_denoise_subtract.py`:

```python
from __future__ import annotations

import unittest

import numpy as np

import sam_audio_utils.denoise as d

SR = 16000


def _tone(freq, seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


class SpectralSubtractTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.clean = _tone(800, 6.0)
        self.noise = 0.05 * rng.standard_normal(len(self.clean))
        self.noisy = (self.clean + self.noise).astype(np.float64)

    def _profile_from_noise(self):
        rng = np.random.default_rng(1)
        noise_only = np.concatenate([
            0.05 * rng.standard_normal(int(4 * SR)),
            self.noisy,
        ])
        return d.estimate_noise_profile(noise_only, SR)[0]

    def test_output_length_and_dtype_preserved(self):
        out = d.spectral_subtract(self.noisy, SR, self._profile_from_noise())
        self.assertEqual(len(out), len(self.noisy))
        self.assertTrue(np.isfinite(out).all(), "no NaN/inf in output")

    def test_noise_is_reduced(self):
        prof = self._profile_from_noise()
        out = d.spectral_subtract(self.noisy, SR, prof)
        before = float(np.sqrt(np.mean((self.noisy - self.clean) ** 2)))
        after = float(np.sqrt(np.mean((out[:len(self.clean)] - self.clean) ** 2)))
        self.assertLess(after, before, "residual noise must go down, not up")

    def test_signal_is_retained(self):
        """Reducing noise must not gut the tone -- correlation stays high."""
        prof = self._profile_from_noise()
        out = d.spectral_subtract(self.noisy, SR, prof)[:len(self.clean)]
        c = float(np.corrcoef(out, self.clean)[0, 1])
        self.assertGreater(c, 0.9, f"signal correlation collapsed to {c:.3f}")

    def test_stronger_setting_removes_more(self):
        prof = self._profile_from_noise()
        gentle = d.spectral_subtract(self.noisy, SR, prof, strength="gentle")
        aggressive = d.spectral_subtract(self.noisy, SR, prof, strength="aggressive")
        e_gentle = float(np.sqrt(np.mean((gentle[:len(self.clean)] - self.clean) ** 2)))
        e_aggr = float(np.sqrt(np.mean((aggressive[:len(self.clean)] - self.clean) ** 2)))
        self.assertLess(e_aggr, e_gentle)

    def test_unknown_strength_rejected(self):
        with self.assertRaises(ValueError):
            d.spectral_subtract(self.noisy, SR, self._profile_from_noise(), strength="nuclear")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_denoise_subtract.py -v`
Expected: FAIL — no attribute `spectral_subtract`

- [ ] **Step 3: Write minimal implementation**

Append to `sam_audio_utils/denoise.py`:

```python
from scipy.signal import istft as _istft


def spectral_subtract(x: np.ndarray, sr: int, profile: np.ndarray,
                      strength: str = "normal") -> np.ndarray:
    """Subtract the noise magnitude spectrum, keeping the original phase.

    alpha over-subtracts to cover frame-to-frame noise fluctuation; beta is a
    spectral floor that stops bins being punched to zero. The floor is what
    prevents musical noise: leaving a low noise bed sounds cleaner than
    removing everything.
    """
    if strength not in STRENGTH_PRESETS:
        raise ValueError(
            f"unknown strength {strength!r}; expected one of {sorted(STRENGTH_PRESETS)}")
    alpha, beta = STRENGTH_PRESETS[strength]

    x = _mono(x)
    n = len(x)
    _, _, Z = _stft(x, fs=sr, nperseg=STFT_WINDOW, noverlap=STFT_OVERLAP)
    mag, phase = np.abs(Z), np.angle(Z)

    noise = profile[:, None]
    # Gain form rather than direct subtraction: it is what the floor clamps.
    gain = 1.0 - alpha * noise / np.maximum(mag, _FLOOR)
    gain = np.maximum(gain, beta)

    # Smooth the gain across adjacent frames: isolated single-frame spikes are
    # exactly what warbles.
    if gain.shape[1] >= 3:
        gain[:, 1:-1] = (gain[:, :-2] + gain[:, 1:-1] + gain[:, 2:]) / 3.0

    _, out = _istft(gain * mag * np.exp(1j * phase), fs=sr,
                    nperseg=STFT_WINDOW, noverlap=STFT_OVERLAP)
    out = np.asarray(out, dtype=np.float64)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out[:n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_denoise_subtract.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add sam_audio_utils/denoise.py tests/test_denoise_subtract.py
git commit -m "feat: spectral subtraction with over-subtraction and spectral floor"
```

---

## Task 4: `denoise_file` — silence passthrough and the target/residual pair

**Files:**
- Modify: `sam_audio_utils/denoise.py`
- Test: `tests/test_denoise_file.py`

**Interfaces:**
- Consumes: everything above.
- Produces (relied on by Task 5):
  - `denoise_file(input_path, target_path, residual_path, strength="normal", progress_cb=None) -> dict`
    Writes both wavs at the INPUT sample rate. Returns a metadata dict with keys
    `method` (always `"spectral_subtraction"`), `strength`, `sample_rate`,
    `duration_seconds`, `noise_window_start_s`, `noise_window_dbfs`,
    `headroom_db`, `drift_db`, `silence_fraction`, `warnings`.
    `progress_cb(pct: int, message: str)` is optional.

- [ ] **Step 1: Write the failing test**

Create `tests/test_denoise_file.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import sam_audio_utils.denoise as d

SR = 44100


class DenoiseFileTests(unittest.TestCase):
    def _make(self, td: Path) -> Path:
        rng = np.random.default_rng(0)
        t = np.arange(int(5 * SR)) / SR
        speech = 0.2 * np.sin(2 * np.pi * 600 * t) + 0.01 * rng.standard_normal(len(t))
        noise = 0.01 * rng.standard_normal(int(4 * SR))
        silence = np.zeros(int(3 * SR))
        x = np.concatenate([speech, silence, noise, speech]).astype(np.float32)
        p = td / "in.wav"
        sf.write(p, x, SR)
        return p

    def test_writes_both_stems_at_input_sample_rate(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._make(td)
            meta = d.denoise_file(src, td / "t.wav", td / "r.wav")
            for name in ("t.wav", "r.wav"):
                data, sr = sf.read(td / name)
                self.assertEqual(sr, SR, "sample rate must be preserved")
                self.assertEqual(len(data), len(sf.read(src)[0]))
            self.assertEqual(meta["method"], "spectral_subtraction")
            self.assertEqual(meta["sample_rate"], SR)

    def test_true_silence_passes_through_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._make(td)
            d.denoise_file(src, td / "t.wav", td / "r.wav")
            out, _ = sf.read(td / "t.wav")
            sil = out[int(5 * SR) + 2000:int(8 * SR) - 2000]  # inside the zero run
            self.assertLess(float(np.abs(sil).max()), 1e-4,
                            "digital silence must stay silent, not gain artifacts")

    def test_target_plus_residual_reconstructs_the_input(self):
        """residual is what was REMOVED -- the pair must add back up."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._make(td)
            d.denoise_file(src, td / "t.wav", td / "r.wav")
            x = sf.read(src)[0].astype(np.float64)
            t = sf.read(td / "t.wav")[0].astype(np.float64)
            r = sf.read(td / "r.wav")[0].astype(np.float64)
            self.assertLess(float(np.abs((t + r) - x).max()), 1e-3)

    def test_progress_callback_is_called(self):
        seen = []
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = self._make(td)
            d.denoise_file(src, td / "t.wav", td / "r.wav",
                           progress_cb=lambda p, m: seen.append(p))
        self.assertTrue(seen, "progress_cb must be called at least once")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_denoise_file.py -v`
Expected: FAIL — no attribute `denoise_file`

- [ ] **Step 3: Write minimal implementation**

Append to `sam_audio_utils/denoise.py`:

```python
from pathlib import Path
from typing import Callable, Optional

import soundfile as sf


def denoise_file(input_path, target_path, residual_path,
                 strength: str = "normal",
                 progress_cb: Optional[Callable[[int, str], None]] = None) -> dict:
    """Denoise a file, writing the cleaned target and the removed residual.

    residual = input - target, so a caller can verify nothing was eaten.
    """
    def report(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    input_path = Path(input_path)
    data, sr = sf.read(input_path, always_2d=False)
    x = _mono(data)

    report(10, "analysing noise floor")
    profile, info = estimate_noise_profile(x, sr)

    report(35, f"subtracting noise (strength={strength})")
    target = spectral_subtract(x, sr, profile, strength=strength)

    # Digital silence stays digitally silent -- overlap-add can otherwise smear
    # energy into a run that was already perfectly clean.
    report(75, "restoring silent regions")
    frame_s = 0.25
    sil = silence_mask(x, sr, frame_s)
    n = max(1, int(frame_s * sr))
    for i, is_sil in enumerate(sil):
        if is_sil:
            target[i * n:(i + 1) * n] = 0.0

    residual = x - target

    report(90, "writing outputs")
    sf.write(Path(target_path), target, sr)
    sf.write(Path(residual_path), residual, sr)

    return {
        "method": "spectral_subtraction",
        "strength": strength,
        "sample_rate": int(sr),
        "duration_seconds": round(len(x) / sr, 3),
        "noise_window_start_s": info["window_start_s"],
        "noise_window_dbfs": info["window_dbfs"],
        "headroom_db": info["headroom_db"],
        "drift_db": info["drift_db"],
        "silence_fraction": round(float(sil.mean()), 4),
        "warnings": info["warnings"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_denoise_file.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full suite — no regressions**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 101 baseline + all new tests passing, still exactly 6 known warnings.

- [ ] **Step 6: Commit**

```bash
git add sam_audio_utils/denoise.py tests/test_denoise_file.py
git commit -m "feat: denoise_file with silence passthrough and target/residual pair"
```

---

## Task 5: Validate against the real file and Paul's WavePad oracle

This is the task that decides whether the tuning is right. It adds a test that measures
real output against a known-good reference, so alpha/beta are tuned against a target
rather than by ear.

**Files:**
- Test: `tests/test_denoise_real_audio.py`
- Modify: `sam_audio_utils/denoise.py` (only if the metrics demand a tuning change)

**Interfaces:**
- Consumes: `denoise_file`, and `_speech_band_fraction` from `worker.handlers.sam_audio_cleanup`.

- [ ] **Step 1: Write the measurement test**

Create `tests/test_denoise_real_audio.py`:

```python
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
```

- [ ] **Step 2: Run it and record every metric**

Run: `.venv/bin/python -m pytest tests/test_denoise_real_audio.py -v`

Report the ACTUAL numbers for all five tests, whether they pass or fail. Do not tune
anything yet — the first run is the measurement.

- [ ] **Step 3: Tune only if a metric fails**

If `test_speech_is_retained` or `test_correlates_with_the_wavepad_oracle` fails, the
default is too aggressive: lower `alpha` and/or raise `beta` in
`STRENGTH_PRESETS["normal"]`. If `test_noise_floor_is_reduced` fails, it is too gentle:
raise `alpha`. Change **one value at a time**, re-run, and record each attempt with its
numbers. Do NOT weaken a threshold to make a test pass — if a metric cannot be met,
stop and report with the evidence.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green, still exactly 6 known warnings.

- [ ] **Step 5: Commit**

```bash
git add tests/test_denoise_real_audio.py sam_audio_utils/denoise.py
git commit -m "test: validate the denoiser against the real file and the WavePad oracle"
```

---

## Task 6: Route it in `clean_cli`

**Files:**
- Modify: `clean_cli.py`
- Test: `tests/test_clean_cli_routing.py`

**Interfaces:**
- Consumes: `denoise_file` from Task 4.
- Produces: `choose_method(description: str, explicit: str = "auto") -> str` returning `"denoise"` or `"separate"`.
- The `job.json` payload gains two optional keys: `method` (`"auto"|"denoise"|"separate"`, default `"auto"`) and `strength` (default `"normal"`). **No new CLI flags** — the bridge and Cortex send what they already send.

- [ ] **Step 1: Write the failing test**

Create `tests/test_clean_cli_routing.py`:

```python
from __future__ import annotations

import unittest

import clean_cli


class ChooseMethodTests(unittest.TestCase):
    def test_cleanup_language_routes_to_denoise(self):
        for desc in ["clean this up", "remove the background noise",
                     "improve poor audio", "a person speaking", "speech", ""]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "denoise")

    def test_named_source_in_a_mixture_routes_to_separate(self):
        for desc in ["the guitar", "a man speaking over a radio",
                     "the voice over the music", "a dog barking"]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "separate")

    def test_explicit_method_always_wins(self):
        self.assertEqual(clean_cli.choose_method("the guitar", "denoise"), "denoise")
        self.assertEqual(clean_cli.choose_method("clean this up", "separate"), "separate")

    def test_unknown_explicit_method_rejected(self):
        with self.assertRaises(ValueError):
            clean_cli.choose_method("anything", "magic")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_clean_cli_routing.py -v`
Expected: FAIL — no attribute `choose_method`

- [ ] **Step 3: Implement `choose_method`**

Add to `clean_cli.py` above `main()`:

```python
# Positive evidence that a SECOND source should be separated out. Anything else
# denoises, because that is the reversible failure: an under-cleaned file is
# still listenable, whereas SAM on single-speaker noise returns an empty stem.
_MIXTURE_WORDS = (
    "guitar", "piano", "drum", "bass", "violin", "music", "song", "instrument",
    "dog", "bark", "bird", "engine", "traffic", "siren", "alarm", "applause",
    "crowd", "tv", "television", "radio", "phone", "typing", "keyboard",
)
_MIXTURE_PHRASES = (" over ", " behind ", " through ", " on top of ", " against ")


def choose_method(description: str, explicit: str = "auto") -> str:
    """Pick the processing method. Explicit always beats inference."""
    if explicit in ("denoise", "separate"):
        return explicit
    if explicit != "auto":
        raise ValueError(
            f"unknown method {explicit!r}; expected auto, denoise or separate")
    text = (description or "").lower()
    if any(w in text for w in _MIXTURE_WORDS):
        return "separate"
    if any(p in f" {text} " for p in _MIXTURE_PHRASES):
        return "separate"
    return "denoise"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_clean_cli_routing.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Wire the denoise path into `main()`**

In `clean_cli.py`, replace the `with gpu_lock(...)` block so the denoise path never
takes the lock. The existing block is:

```python
        payload = json.loads(Path(args.job_json).read_text())
        with gpu_lock(args.gpu_lock_timeout):
            result = handle(...)
```

Replace with:

```python
        payload = json.loads(Path(args.job_json).read_text())
        method = choose_method(payload.get("description", ""),
                               payload.get("method", "auto"))
        strength = payload.get("strength", "normal")
        emit(5, "route", f"method={method}")

        if method == "denoise":
            # No GPU lock, no model: this path is pure CPU DSP, so it runs even
            # while a SAM job holds the card.
            work = out_dir / "denoise_work"
            work.mkdir(parents=True, exist_ok=True)
            target_wav = work / "target.wav"
            residual_wav = work / "residual.wav"
            meta = denoise_file(input_path, target_wav, residual_wav,
                                strength=strength,
                                progress_cb=lambda p, m: emit(p, "denoise", m))
            meta["description"] = payload.get("description", "")
            meta["input_filename"] = input_path.name
            zip_src = work / "result.zip"
            with zipfile.ZipFile(zip_src, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(target_wav, "target.wav")
                zf.write(residual_wav, "residual.wav")
                zf.writestr("metadata.json", json.dumps(meta, indent=2))
            result = {"output_data": meta, "output_file": zip_src}
        else:
            with gpu_lock(args.gpu_lock_timeout):
                result = handle(
                    input_path=input_path,
                    input_data=payload,
                    job={"id": 0, "input_filename": input_path.name},
                    progress_cb=lambda pct, msg, stage=None: emit(pct, stage or "processing", msg),
                    is_cancelled_cb=cancel.is_set,
                    work_dir=out_dir,
                )
```

Add the import near the existing handler import:

```python
from sam_audio_utils.denoise import denoise_file  # noqa: E402
```

- [ ] **Step 6: Verify the contract is unchanged with a real end-to-end CLI run**

```bash
SCRATCH=/tmp/denoise_smoke && mkdir -p "$SCRATCH"
printf '{"description": "clean this up"}' > "$SCRATCH/job.json"
.venv/bin/python clean_cli.py --input Apollo13.wav --job-json "$SCRATCH/job.json" \
  --out-dir "$SCRATCH/out" --opus
echo "rc=$?"
cat "$SCRATCH/out/status.json"
unzip -l "$SCRATCH/out/result.zip"
ffprobe -v error -show_entries stream=codec_name,channels "$SCRATCH/out/target.ogg"
```

Expected: `rc=0`; `status.json` state `done`; `result.zip` contains target.wav,
residual.wav, metadata.json; `target.ogg` is Opus mono; `metadata.json` records
`"method": "spectral_subtraction"`. Report the real output.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green, exactly 6 known warnings.

- [ ] **Step 8: Commit**

```bash
git add clean_cli.py tests/test_clean_cli_routing.py
git commit -m "feat: route cleanup jobs to the denoiser by default, SAM on request"
```

---

## Task 7: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the denoiser**

Add a `## Denoising (default cleanup path)` section to `README.md` after the existing
`## Discord loop (Hermes)` section, covering: that denoising is now the default and SAM
runs only on positive evidence of a mixture; the `method` and `strength` keys in
`job.json`; that the denoise path uses no GPU and takes no lock; the silence gate and
why it exists (a collation of cuts contains true digital zero, and a profile taken from
it subtracts nothing); and that `residual.wav` is how you check the voice was not eaten.

Match the existing tone and heading style. Do not restructure the rest of the file.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: spectral denoiser as the default cleanup path"
```

---

## Self-review notes (author)

- **Spec coverage:** Component 1 (silence gate + profile + guards) → Tasks 1-2;
  Component 2 (subtraction, alpha/beta, smoothing, sample-rate preservation) → Task 3;
  Component 3 (module, routing, contract, no GPU lock) → Tasks 4, 6; Component 4
  (validation: speech-band, oracle correlation, noise-floor reduction, silence
  passthrough) → Task 5. Non-goals are not implemented anywhere. All mapped.
- **Type consistency:** `estimate_noise_profile` returns `(profile, info)` in Task 2 and
  is consumed that way in Task 4; `spectral_subtract(x, sr, profile, strength)` matches
  between Tasks 3 and 4; `denoise_file(...) -> dict` matches between Tasks 4, 5 and 6;
  `choose_method(description, explicit)` matches between Tasks 6's test and
  implementation.
- **Deliberate deviation from the spec, flagged:** the spec says the noise window is the
  quietest window among non-silent frames. Task 2 implements the stricter rule that the
  window must contain **no** silent frame at all, rather than merely averaging over
  survivors. Averaging across a window straddling a cut boundary would pull the profile
  toward zero — the exact failure the gate exists to prevent. Same intent, tighter rule.
- **Known risk left to the implementer:** Task 5's thresholds (speech-band > 0.30,
  oracle correlation > 0.70, floor reduction > 3 dB) are first estimates. They are
  deliberately loose enough to pass a decent implementation and tight enough to fail the
  SAM-style catastrophe. Task 5 Step 3 forbids weakening them to force a pass.
