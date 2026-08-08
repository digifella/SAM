# "Remove the X" Returns SAM's Residual — Design

**Date:** 2026-08-08
**Status:** approved (design), not yet implemented
**Supersedes:** the routing ruling in commit `40edf52` (removal phrasing → denoise)
**Depends on:** the spectral denoiser (merged, `af959e7`) staying the fallback path

## Why this exists

A user asking to "remove the barking dog" is asking for a stem SAM already computes.
SAM extracts the sound you *describe* and writes it to `target.wav`, putting everything
else in `residual.wav`. For a removal request the residual **is** the deliverable — the
pipeline simply hands back the wrong one of the two files it already produced.

Today that request routes to the denoiser instead. The denoiser keeps the voice safe but
cannot remove a dog, so the user's actual request goes unserved. This closes that gap.

## The ruling this reverses, and why reversing it is safe

Commit `40edf52` sent all removal phrasing to denoise, for a sound reason at the time:

> "remove the barking dog" returns the DOG as target.wav, pushes the voice into
> residual.wav, and reports success — the speech-band guard never arms, because those
> descriptions never mention speech. This project's founding failure, reachable through
> ordinary English.

That reasoning is about handing back SAM's **target**. It does not transfer to handing
back SAM's **residual**, because the two fail in opposite directions:

| SAM finds nothing | returning its target | returning its residual |
|---|---|---|
| result | near-empty stem — voice lost | ≈ the whole input — nothing removed |
| severity | **catastrophic, silent** | benign under-clean, same as denoise |

Returning the residual is the *reversible* direction — the property `40edf52` chose
denoise for in the first place. The handler's own warning text at
`sam_audio_cleanup.py:236` already anticipates this shape of request, naming
"instructions like 'remove X'".

**Consequence, accepted deliberately:** `tests/test_clean_cli_routing.py::RemovalPhrasingMustNotReachSamTests`
encodes the old ruling in its name and docstring, and all 13 of its descriptions flip to
the new route. That class is **replaced**, not edited, by one asserting the inverse with
a docstring recording why the ruling changed. If the table above is wrong, this whole
design is wrong.

## Decisions

1. **Trigger = removal cue + a nameable source.** Not all removal phrasing. "remove
   background noise" and "remove the hiss" keep denoising, because neither names a source
   SAM can extract — asking SAM for "hiss" is the single-speaker case that returns an
   empty stem.
2. **Swap inside `handle()`, not after it.** See "The constraint that drives the design".
3. **Prompt = the description minus the cue.** Deterministic string surgery, no LLM.
   Descriptions naming both what to remove and what to keep ("remove the dog barking over
   her voice") stay ambiguous and send the whole remainder — accepted, see Non-goals.
4. **Two-sided guard.** Check the deliverable *and* what was removed.

## The constraint that drives the design

Everything in `handle()` after the stems resolve is **asymmetric on `target_path`**:

```
653  target_path   = _find_single_output(output_dir, "_target")
654  residual_path = _find_single_output(output_dir, "_residual")
656  _separation_sanity_warning(target_path, residual_path, description)
663  _apply_peak_normalize(target_path, ...)      # both stems — symmetric
670  _apply_loudness_normalize(target_path, ...)  # TARGET ONLY — asymmetric
681  info = sf.info(target_path)                  # metadata from target
713  zf.write(target_path, arcname="target.wav")
```

So swapping the zip members *after* `handle()` returns would ship three quiet bugs:
loudness normalization applied to the removed dog, `metadata.json` describing the wrong
stem, and a guard warning reading backwards. Swapping at line 655 makes every one of
those correct by construction.

This is also why a post-hoc swap in `clean_cli.py` was rejected despite keeping the
handler's signature frozen: it would have to re-run loudnorm and reimplement
`_speech_band_fraction` outside the module that owns it.

## Component 1 — routing

`choose_method()` gains a third return value, `"remove"`, refining the branch that
already runs first:

```python
if any(r in text for r in _REMOVAL_CUES):
    if _MIXTURE_RE.search(text) or (
            _MEDIUM_RE.search(text) and any(c in text for c in _BACKGROUND_CUES)):
        return "remove"
    return "denoise"          # no nameable source — unchanged
```

The medium clause is included so "remove the tv in the background" routes to `remove`;
"remove noise from this phone message" still denoises, having no background cue.

`method` in `job.json` accepts `"remove"` explicitly, and explicit always beats inference.

## Component 2 — prompt derivation

`strip_removal_cue(description) -> str` in `clean_cli.py`: drop the matched removal cue
and any leading article, return the remainder.

```
"remove the barking dog"   -> "barking dog"
"get rid of the guitar"    -> "guitar"
"filter out the siren"     -> "siren"
```

The trigger guarantees a mixture or medium word survives the strip, so the prompt is
never empty. If it is anyway, **fall back to denoise** rather than sending SAM a blank
description — `handle()` defaults an empty description to `"speech"`, which on a removal
job would extract exactly the thing the user wants to keep.

That fallback lives in `clean_cli.py`, after routing and before dispatch: `choose_method()`
stays a pure function of the description, and the downgrade happens where the derived
prompt is computed. So `method="remove"` from `choose_method()` is a *proposal* that an
empty prompt can veto; an explicit `method="remove"` in `job.json` is vetoed the same way,
since a blank prompt is unusable regardless of who asked for it.

## Component 3 — the swap

`handle()` gains `remove_mode: bool = False`. Immediately after line 654:

```python
if remove_mode:
    target_path, residual_path = residual_path, target_path
```

Existing callers are unaffected by the default.

**`residual.wav = input − target.wav` is NOT an invariant on this path.** It is exact on
the denoiser (`sam_audio_utils/denoise.py` literally computes `residual = x - target`),
but SAM does not subtract anything: `run_sam_interactive.py:411-416` unbatches
`wavs[:, 0]` and `wavs[:, 1]` as two *independently generated* codec outputs at the same
rerank index. Both stems are resynthesized, so their sum is not guaranteed to reconstruct
the input, and how close it comes has not been measured here. The swap is still correct —
`residual.wav` is the stem holding everything SAM did not attribute to the named source —
but the benign-failure argument above ("≈ the whole input") is an empirical claim about a
generative model, not algebra.

## Component 4 — two-sided guard

`_separation_sanity_warning()` takes `remove_mode` and, when set:

- **Arms the speech check unconditionally.** The derived prompt is "barking dog", which
  contains no speech word, so description-driven arming would never fire — the precise
  hole that made the founding failure silent.
- **Adds the mirror check:** warn when the removed stem's speech-band fraction both
  exceeds `MIN_SPEECH_BAND_FRACTION` *and* exceeds the deliverable's. Two conditions, not
  one: "merely higher" would fire whenever the discarded sound happens to carry incidental
  mid-band energy, and a bare threshold would fire on a genuinely speech-adjacent source.
  This mirrors the existing comparison at `sam_audio_cleanup.py:256`. It is the founding
  failure inverted, and nothing else would catch it.

The existing near-silent RMS check still runs first, now against the deliverable, where
near-silence means SAM classified nearly the whole file as the sound to remove.

Warning text needs a removal-specific variant: "the voice is probably in residual.wav"
means the opposite thing when residual.wav holds the discarded sound.

## Component 5 — metadata

`metadata.json` gains:

- `method: "remove"`
- `removed: "<derived prompt>"` — what SAM was actually asked to extract

`description` keeps the user's original words, so a listener can see both what was asked
and what was acted on.

## Component 6 — validation

- **Routing:** replace the inverted class. Re-run the existing 42-must-denoise corpus
  **unchanged** — that no removal-without-a-source leaked into `remove` is the regression
  that matters most.
- **`strip_removal_cue()`:** unit test per surface form the cue pattern matches.
- **Swap:** synthetic two-tone mixture; assert `target.wav` is the tone *not* named.
  Do **not** assert that `target + residual` reconstructs the input — see Component 3;
  that holds on the denoiser, not on SAM.
- **Guard:** both new warnings fire on constructed stems; a normal `separate` job is
  unaffected.
- **Real audio:** `Fionnuala_raw.wav` is single-speaker and proves nothing here. A real
  mixture file is needed, and Paul may have to supply one. Until then the swap is
  verified synthetically — stated plainly rather than claimed as end-to-end coverage.

## Non-goals

- **Disambiguating descriptions that name both sides.** "remove the dog barking over her
  voice" sends the whole remainder. Truncating at `_MIXTURE_PHRASES` was considered and
  deferred: it adds a rule to reason about for a phrasing not yet observed in traffic.
- **Removing multiple named sources.** One SAM pass, one named source.
- **Improving SAM's extraction quality.** This routes to a stem SAM already computes.
- **Stereo.** Mono output, consistent with both existing paths (ruled 2026-08-08).

## Risks

- **The trigger fires on a topic rather than a source.** "remove the part about traffic"
  names a subject of speech, not a sound, and would route to `remove`. Inherited from the
  existing routing limitation documented in the README; mitigated by the two-sided guard,
  which should catch the resulting voice loss, and by explicit `method`.
- **SAM extracts the voice as the named source.** The mirror check exists precisely for
  this and warns rather than fails, consistent with how the pipeline treats every other
  sanity finding.
- **The corpus that validated `40edf52` shrinks.** 13 descriptions move from must-denoise
  to must-remove, so the must-denoise corpus loses coverage. Retain all 13 as explicit
  must-remove assertions so the total is unchanged and the boundary stays pinned.
