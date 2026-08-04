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

    # frame_rms_dbfs (and therefore silence_mask) only covers whole frames and
    # drops any trailing remainder shorter than one frame. Check that leftover
    # tail directly -- via the same single-value branch frame_rms_dbfs takes
    # for signals shorter than a frame -- so a silent tail isn't left smeared
    # by overlap-add just because it didn't land on a frame boundary.
    tail_start = (len(x) // n) * n
    tail = x[tail_start:]
    if tail.size and frame_rms_dbfs(tail, sr, frame_s)[0] <= SILENCE_GATE_DBFS:
        target[tail_start:] = 0.0

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
