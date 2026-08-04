# Spectral-Subtraction Denoiser — Design

**Date:** 2026-08-04
**Status:** approved (design), not yet implemented
**Test case:** `Fionnuala_raw.wav` (9m39s, 44.1kHz mono, 24-bit)
**Oracle:** `Fionnuala_raw_spectral subtraction.wav` — Paul's WavePad result, known good

## Why this exists

SAM-Audio is description-driven **source separation**: pull one named sound out of a
mixture of distinct sources. It is not a denoiser. Asked to clean a single speaker in
broadband noise it failed badly and silently — the target it returned was **93.5%
sub-300Hz rumble with 0.1% speech-band energy**, while the speaker's voice went into
the discarded residual (58.6% speech band). The pipeline reported success and posted
the empty stem to Discord.

Paul solved the same file by hand in NCH WavePad using spectral subtraction with a
manually chosen 3-second noise sample. That result correlates **0.978 with the raw
input** — it removed noise without touching the voice.

Most material realistically attached in Discord is this shape: one speaker, background
noise, no separable second source. The denoiser should therefore be the default path
and SAM the special case.

A speech-band guard was added to the existing pipeline separately (commit `86568c1`) so
the SAM path can no longer report success on an output containing no speech. That guard
is a safety net, not a substitute for this tool.

## Decisions

| Decision | Choice |
|---|---|
| Method routing | Denoise by default; SAM only when the description names a source to extract |
| Noise estimation | Auto-pick the quietest window, **among non-silent frames only** |
| Strength | One conservative tuned default + optional gentle/normal/aggressive knob |
| Architecture | New `sam_audio_utils/denoise.py`, routed inside `clean_cli.py` |

## The constraint that drives the design

The test file is a **collation of separate audio cuts**, so parts of it are true digital
silence, not quiet noise. Measured:

```
2315 frames of 250ms
frame dBFS: min -240.0  p1 -240.0  p5 -240.0  p10 -240.0  median -30.9  max -8.4
TRUE-SILENCE frames (< -90 dBFS): 579 / 2315 = 25.0%
absolute-zero frames (max|x| == 0): 577
13 silence runs, 1.25s to 28.00s
```

**The bottom 10% of the file is digital zero.** Both obvious noise estimators are
therefore dead on arrival:

- *quietest window* → lands in a zero run → profile is all zeros → subtracts nothing
- *per-bin percentile (p10)* → also −240 dBFS → same

Either would produce a silent no-op that looks like success. This is the same class of
failure as the SAM incident, so it must be designed out, not tested for later.

Once zeros are excluded, the noise is **stationary**. Quiet-frame floor (p10 of
non-silent frames) per 60s block:

```
-37.4  -39.4  -39.5  -37.0  -37.4  -39.2  -39.2  -39.8  -40.2  -38.2  dBFS
```

A ~3 dB spread over 9½ minutes. One global noise profile is sufficient; per-cut
profiles and adaptive minimum-statistics tracking are **not built** (YAGNI).

## Component 1 — noise estimation

1. **Gate true silence.** Classify a frame as digital silence if `max|x| == 0` or its
   RMS is below `SILENCE_GATE_DBFS = -90`. Exclude these from *all* noise statistics.
2. **Pick the quietest window among survivors.** Slide a `NOISE_WINDOW_SECONDS = 3.0`
   window over non-silent frames; choose the lowest-energy position. Average its
   magnitude spectrum per frequency bin to form the profile.
3. **Validate the pick before use.** Two guards, both fatal-with-a-clear-error rather
   than silently proceeding:
   - the window's level must be **above** the silence gate — otherwise we are still in
     a zero run and the profile is empty;
   - it must be **meaningfully below the file's speech median** (≥ 6 dB below).
     Otherwise the "quietest" window contains speech and subtracting it removes the
     speaker from herself.
4. **Measure noise drift.** Compute the per-block quiet floor; if the spread exceeds
   `MAX_NOISE_DRIFT_DB = 10`, warn that one global profile may not fit — do not silently
   misapply it.
5. **Pass true-silence regions through untouched.** They are already clean; processing
   them can only add artifacts.

## Component 2 — subtraction

STFT via `scipy.signal.stft`: 2048-sample window (≈46ms at 44.1kHz), 75% overlap.
Per frame, subtract the noise magnitude, retain the **original phase**, inverse-STFT
with overlap-add.

Two parameters govern musical noise:

- **α, over-subtraction factor** — subtract `α ×` the estimate. Compensates for the
  profile being an average of fluctuating noise. Higher α is quieter but warblier.
- **β, spectral floor** — never attenuate a bin below `β ×` its original magnitude.
  This is the anti-musical-noise mechanism: it leaves a low noise bed instead of
  punching isolated spectral holes, which is what warbles. Leaving *some* noise sounds
  cleaner than removing all of it.

| strength | α | β | character |
|---|---|---|---|
| gentle | 1.5 | 0.10 (−20dB) | audible reduction, no artifacts |
| **normal** (default) | 2.0 | 0.05 (−26dB) | tuned target |
| aggressive | 3.0 | 0.02 (−34dB) | quieter, risks warble |

Plus **gain smoothing across adjacent time frames**, suppressing the isolated
single-frame spikes that cause warble.

**Sample rate is preserved** — 44.1kHz in, 44.1kHz out. No resampling (the SAM path
forces 48k; this one must not).

## Component 3 — integration

`sam_audio_utils/denoise.py` is a standalone module: **numpy/scipy only, no torch, no
GPU, no model**. Runs faster than realtime on CPU.

`clean_cli.py` routes: denoise by default, SAM when the description names a source to
extract. The denoise path **skips `gpu_lock` entirely**, so a denoise job runs
immediately even while a SAM job holds the card.

### How routing decides

Both new inputs arrive in the existing `job.json` payload — **no new CLI flags**, so the
bridge and Cortex keep sending exactly what they send today:

```json
{"description": "...", "method": "auto|denoise|separate", "strength": "gentle|normal|aggressive"}
```

`method` defaults to `"auto"`, `strength` to `"normal"`. Explicit values always win over
inference, which keeps the behaviour testable and gives Hermes an escape hatch.

Under `"auto"`, the rule is deliberately conservative — **denoise unless there is
positive evidence a second source should be separated out**:

- Route to **separate** when the description names a source to isolate *and* implies a
  mixture: an instrument, a named non-speech sound, or speech qualified by a competing
  source ("the man over the radio", "the guitar", "the voice over the music").
- Route to **denoise** otherwise, including the whole cleanup family — "clean this up",
  "remove background noise", "improve poor audio", "a person speaking", or an empty
  description.

Ambiguity resolves to denoise, because that is the reversible failure: an
under-cleaned file is still listenable, whereas SAM on single-speaker noise returns an
empty stem. The chosen method and the reason are written to `metadata.json` and stated
in the Discord reply.

**The output contract is unchanged**: `result.zip` containing `target.wav` (denoised),
`residual.wav` (what was removed), `metadata.json`; plus `status.json`, `--opus`, and
the Discord webhook exactly as now. Cortex, the bridge, the sp4 wrapper and Hermes
require **zero changes**.

`residual.wav` is not decorative — it is how a caller verifies the tool did not eat the
voice, which is precisely how the SAM failure was diagnosed.

The method actually used is recorded in `metadata.json` and stated in the Discord reply,
so it is never ambiguous which ran.

## Component 4 — validation

Automated, against Paul's WavePad output as oracle. These are regression tests, not
listening sessions — they are what allow α and β to be tuned against a target.

1. **Speech-band retention.** The denoised target must retain speech-band (300–3400Hz)
   energy. Anything near the 0.1% of the SAM failure is an automatic fail. Reuses
   `_speech_band_fraction` from `worker/handlers/sam_audio_cleanup.py`.
2. **Correlation vs the WavePad oracle.** Paul's result sits at 0.978 against the raw
   input. Materially below that means speech is being damaged; far above means nothing
   is being removed.
3. **Noise-floor reduction**, measured in quiet non-silent regions against the ~−38 dBFS
   starting floor.
4. **True-silence regions come out bit-identical**, since they are passed through.

## Non-goals

- Per-cut noise profiles (drift is 3 dB — not warranted)
- Minimum-statistics adaptive tracking (same reason)
- Replacing SAM — it stays, for genuine multi-source separation
- Repairing clipping. The source has 144 full-scale samples; those are destroyed and no
  denoiser reconstructs them.

## Risks

| Risk | Mitigation |
|---|---|
| Noise profile picked from digital silence → silent no-op | Silence gate + explicit post-pick validation (Component 1.3) |
| Noise window contains speech → subtracts the voice | 6 dB-below-speech-median guard |
| Musical noise / warble | Spectral floor β, gain smoothing, conservative default |
| Non-stationary noise on a future file | Drift measurement + warning |
| Tuning to one file | Oracle-based metrics, plus Apollo13 as a second, different test input |
