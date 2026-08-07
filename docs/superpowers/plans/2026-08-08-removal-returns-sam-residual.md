# "Remove the X" Returns SAM's Residual — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `"remove the barking dog"` return the audio *without* the dog, by running SAM and handing back its residual instead of its target.

**Architecture:** SAM already computes both stems. A new `remove` route derives an extraction prompt from the removal phrasing, runs the normal separation path, and swaps `target_path`/`residual_path` inside `handle()` immediately after they resolve — so the sanity guard, loudness normalization, `sf.info` metadata and the ZIP all act on the correct stem by construction. The guard becomes two-sided because a removal description never contains a speech word.

**Tech Stack:** Python 3.11, `unittest` (not pytest-style asserts — this repo uses `unittest.TestCase` with `subTest`), numpy, soundfile. Run the suite with `.venv/bin/python -m pytest tests/ -q`.

## Global Constraints

- **Run tests in the FOREGROUND. Never `git add -A`.** Three files must stay untracked: `Apollo13.wav:Zone.Identifier`, `Fionnuala_raw.wav:Zone.Identifier`, `sam-audio-colab-upload.zip`.
- **Baseline is 148 passed, exactly 6 known third-party warnings** (pynvml, pkg_resources ×2, torch weight_norm, SWIG ×2), 73 subtests. Any change to the warning count is a regression.
- **Do NOT add `-ed`/`-ing` to `_MIXTURE_RE`.** It reads "alarm" out of "alarming" and "crowd" out of "crowded". This was a real bug, fixed twice.
- **`MIN_SPEECH_BAND_FRACTION = 0.10` and `SPEECH_BAND_HZ = (300.0, 3400.0)` are unchanged.** Do not retune them to make a test pass.
- Existing callers of `handle()` must keep working untouched — every new parameter defaults to the current behaviour.
- Spec: `docs/superpowers/specs/2026-08-08-removal-returns-sam-residual-design.md`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `clean_cli.py` | Routing + prompt derivation + dispatch | Modify: `choose_method()`, new `strip_removal_cue()`, dispatch block |
| `worker/handlers/sam_audio_cleanup.py` | Separation, stem orientation, guard, metadata | Modify: `_separation_sanity_warning()`, `handle()` |
| `tests/test_clean_cli_routing.py` | Routing contract | Modify: replace `RemovalPhrasingMustNotReachSamTests` |
| `tests/test_strip_removal_cue.py` | Prompt derivation | Create |
| `tests/test_remove_mode_swap.py` | Stem orientation + metadata | Create |
| `tests/test_removal_guard.py` | Two-sided guard | Create |
| `README.md` | User-facing contract | Modify: routing section |

---

### Task 1: `strip_removal_cue()` — derive SAM's prompt

**Files:**
- Modify: `clean_cli.py` (add after `_REMOVAL_CUES`, around line 174)
- Test: `tests/test_strip_removal_cue.py` (create)

**Interfaces:**
- Consumes: `_REMOVAL_CUES` (existing tuple in `clean_cli.py`)
- Produces: `clean_cli.strip_removal_cue(description: str) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_strip_removal_cue.py`:

```python
"""The SAM prompt for a removal job is the description minus the removal cue.

Sending the raw description would hand SAM "remove the barking dog" as the sound
to extract, which is the bug this whole route exists to fix.
"""
import unittest

import clean_cli


class StripRemovalCueTests(unittest.TestCase):
    def test_strips_cue_and_leading_article(self):
        cases = {
            "remove the barking dog": "barking dog",
            "get rid of the guitar": "guitar",
            "filter out the siren": "siren",
            "take out the traffic noise": "traffic noise",
            "cut out the keyboard typing": "keyboard typing",
            "suppress the crowd noise": "crowd noise",
            "eliminate the applause": "applause",
            "strip out the piano": "piano",
            "delete the dog barking": "dog barking",
            "mute the typing": "typing",
        }
        for desc, expected in cases.items():
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.strip_removal_cue(desc), expected)

    def test_every_cue_is_strippable(self):
        """No cue may survive into the prompt -- that is the whole point."""
        for cue in clean_cli._REMOVAL_CUES:
            desc = f"{cue} the guitar"
            with self.subTest(cue=cue):
                self.assertEqual(clean_cli.strip_removal_cue(desc), "guitar")

    def test_mid_sentence_cue_leaves_the_source_named(self):
        """Awkward but accepted: the remainder is not re-flowed as English.

        The prompt is clumsy; what matters is that the mixture word survives so
        SAM still has the source to extract. Documented in the spec's non-goals.
        """
        out = clean_cli.strip_removal_cue(
            "there is an alarm going off in the background, please remove it")
        self.assertIn("alarm", out)
        self.assertNotIn("remove", out)

    def test_empty_and_cue_only_return_empty(self):
        """An empty prompt vetoes the route (Task 5) rather than reaching SAM."""
        for desc in ["", "   ", "remove", "get rid of", "remove the"]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.strip_removal_cue(desc), "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_strip_removal_cue.py -q`
Expected: FAIL with `AttributeError: module 'clean_cli' has no attribute 'strip_removal_cue'`

- [ ] **Step 3: Write minimal implementation**

In `clean_cli.py`, immediately after the `_REMOVAL_CUES` tuple:

```python
# Leading determiners left behind once the cue is gone: "remove the guitar"
# strips to "the guitar", and SAM does better with the bare source name.
_LEADING_ARTICLE_RE = re.compile(
    r"^(?:the|a|an|any|all|that|this|those|these)\s+", re.IGNORECASE)


def strip_removal_cue(description: str) -> str:
    """Derive SAM's extraction prompt from a removal request.

    SAM extracts the sound it is DESCRIBED, so the cue must not survive:
    "remove the barking dog" has to reach SAM as "barking dog". The remainder
    is not re-flowed as English -- a cue in the middle of a sentence leaves a
    clumsy prompt, which is accepted so long as the named source survives.

    Returns "" when nothing usable is left; the caller vetoes the route.
    """
    text = (description or "").strip()
    if not text:
        return ""
    low = text.lower()
    # Longest cue first: "take out" must win over any shorter cue it contains.
    for cue in sorted(_REMOVAL_CUES, key=len, reverse=True):
        i = low.find(cue)
        if i == -1:
            continue
        text = f"{text[:i]} {text[i + len(cue):]}".strip()
        break
    text = _LEADING_ARTICLE_RE.sub("", text.strip(), count=1)
    return " ".join(text.split())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_strip_removal_cue.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add clean_cli.py tests/test_strip_removal_cue.py
git commit -m "feat: derive SAM's extraction prompt from removal phrasing"
```

---

### Task 2: Routing — `choose_method()` gains `"remove"`

**Files:**
- Modify: `clean_cli.py:176-193` (`choose_method`)
- Test: `tests/test_clean_cli_routing.py:98-126` (replace the class)

**Interfaces:**
- Consumes: `_REMOVAL_CUES`, `_MIXTURE_RE`, `_MEDIUM_RE`, `_BACKGROUND_CUES` (all existing)
- Produces: `choose_method()` returns one of `"denoise" | "separate" | "remove"`

**This task inverts a documented ruling.** `RemovalPhrasingMustNotReachSamTests` asserts removal phrasing denoises; all 13 of its descriptions now route to `remove`. Replace the class wholesale — do not edit assertions one by one, because the class name and docstring both encode the old ruling.

- [ ] **Step 1: Write the failing test**

In `tests/test_clean_cli_routing.py`, replace the entire class at lines 98-126 (keep `test_naming_a_source_to_keep_still_separates` at line 128 — it is a different test and still valid):

```python
class RemovalPhrasingReturnsTheResidualTests(unittest.TestCase):
    """"Remove the X" runs SAM on X and hands back the RESIDUAL.

    This inverts the ruling in 40edf52, which sent all removal phrasing to
    denoise. That reasoning was about returning SAM's TARGET: when SAM finds
    nothing, the target is a near-empty stem and the voice is lost silently.
    It does not transfer to returning SAM's RESIDUAL -- when SAM finds nothing
    the residual is approximately the whole input, so the failure is a benign
    under-clean, the same reversible direction 40edf52 chose denoise for.

    All 13 descriptions below were must-denoise assertions under the old
    ruling. They are retained here as must-remove so the boundary stays pinned
    and the corpus does not shrink.
    """

    def test_removal_of_a_named_source_routes_to_remove(self):
        for desc in [
            "remove the barking dog",
            "take out the traffic noise",
            "cut out the keyboard typing",
            "remove the engine noise from this recording",
            "suppress the crowd noise",
            "there is an alarm going off in the background, please remove it",
            "get rid of the music",
            "filter out the siren",
            "eliminate the applause",
            "strip out the piano",
            "mute the typing",
            "delete the dog barking",
            "remove the tv in the background",
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "remove")

    def test_removal_without_a_named_source_still_denoises(self):
        """The boundary that must NOT move: hiss and noise are not sources.

        Asking SAM to extract "hiss" is the single-speaker case that returns an
        empty stem. These stay on the denoiser.
        """
        for desc in [
            "remove the background noise",
            "remove the noise in the background",
            "remove the hiss",
            "get rid of the static",
            "remove noise from this phone message",
            "clean this up and remove the hum",
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "denoise")

    def test_explicit_method_still_wins(self):
        self.assertEqual(
            clean_cli.choose_method("remove the barking dog", "denoise"), "denoise")
        self.assertEqual(
            clean_cli.choose_method("clean this up", "remove"), "remove")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_clean_cli_routing.py -q`
Expected: FAIL — `AssertionError: 'denoise' != 'remove'` on the first subtest, and a `ValueError` on the explicit `"remove"` case.

- [ ] **Step 3: Write minimal implementation**

Replace `choose_method()` in `clean_cli.py` (lines 176-193):

```python
def choose_method(description: str, explicit: str = "auto") -> str:
    """Pick the processing method. Explicit always beats inference."""
    if explicit in ("denoise", "separate", "remove"):
        return explicit
    if explicit != "auto":
        raise ValueError(
            f"unknown method {explicit!r}; expected auto, denoise, separate or remove")
    text = (description or "").lower()
    # Checked FIRST: a removal request names the sound to DISCARD. If that sound
    # is one SAM can extract, run SAM and hand back the residual; if it is only
    # "noise" or "hiss", there is no source to extract, so denoise instead.
    if any(r in text for r in _REMOVAL_CUES):
        if _MIXTURE_RE.search(text):
            return "remove"
        if _MEDIUM_RE.search(text) and any(c in text for c in _BACKGROUND_CUES):
            return "remove"
        return "denoise"
    if _MIXTURE_RE.search(text):
        return "separate"
    if any(p in f" {text} " for p in _MIXTURE_PHRASES):
        return "separate"
    if _MEDIUM_RE.search(text) and any(c in text for c in _BACKGROUND_CUES):
        return "separate"
    return "denoise"
```

- [ ] **Step 4: Run the FULL routing suite to verify the 42-description corpus is intact**

Run: `.venv/bin/python -m pytest tests/test_clean_cli_routing.py -q`
Expected: PASS. **The must-denoise corpus elsewhere in this file must pass unchanged** — if any of those 42 descriptions now returns `"remove"`, a removal-without-a-source leaked into the new route. That is the regression that matters most; fix the routing, never the corpus.

- [ ] **Step 5: Commit**

```bash
git add clean_cli.py tests/test_clean_cli_routing.py
git commit -m "feat: route removal of a named source to the new remove method"
```

---

### Task 3: Two-sided sanity guard

**Files:**
- Modify: `worker/handlers/sam_audio_cleanup.py:219-266` (`_separation_sanity_warning`)
- Test: `tests/test_removal_guard.py` (create)

**Interfaces:**
- Consumes: `_speech_band_fraction`, `_rms_dbfs`, `MIN_SPEECH_BAND_FRACTION`, `SILENT_RMS_DBFS`, `SPEECH_BAND_HZ` (all existing)
- Produces: `_separation_sanity_warning(target_path, residual_path, description, remove_mode: bool = False) -> Optional[str]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_removal_guard.py`:

```python
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
        self.assertIn("removed", msg.lower())
        # The old wording points at the wrong file on a removal job.
        self.assertNotIn("voice is probably THERE", msg)

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
        self.assertIn("near-silent", msg.lower())

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_removal_guard.py -q`
Expected: FAIL with `TypeError: _separation_sanity_warning() got an unexpected keyword argument 'remove_mode'`

- [ ] **Step 3: Write minimal implementation**

Replace `_separation_sanity_warning` in `worker/handlers/sam_audio_cleanup.py`:

```python
def _separation_sanity_warning(target_path: Path, residual_path: Path, description: str,
                               remove_mode: bool = False) -> Optional[str]:
    """Flag a target that cannot be what the description asked for.

    Two failure modes, both seen in practice:
      1. near-silent target -- the description matched nothing;
      2. loud target with no speech in it -- the description asked for a voice
         but SAM extracted rumble, and the voice went into the residual. RMS
         cannot see this one, which is why it shipped undetected.

    On a removal job the stems arrive swapped: target_path is what the user
    KEEPS and residual_path is what was removed. Two things change. The speech
    check arms unconditionally, because the derived prompt ("barking dog")
    never contains a speech word and description-driven arming would never
    fire. And a third failure mode becomes checkable -- the voice ending up in
    what was DISCARDED, which is failure mode 2 inverted.
    """
    target_db = _rms_dbfs(target_path)
    residual_db = _rms_dbfs(residual_path)

    if target_db < SILENT_RMS_DBFS:
        if remove_mode:
            return (
                f"Almost nothing is left after removing '{description}' - the kept audio "
                f"is near-silent ({target_db:.1f} dBFS RMS). SAM classified nearly the "
                f"whole recording as the sound to remove, so the description is very "
                f"likely too broad. What was removed is in residual.wav "
                f"({residual_db:.1f} dBFS RMS)."
            )
        return (
            f"Target output is near-silent ({target_db:.1f} dBFS RMS) - the description "
            f"'{description}' may not match any sound in the audio. SAM-Audio extracts the "
            f"sound you DESCRIBE (e.g. 'a man speaking over a radio'), it does not follow "
            f"instructions like 'remove X'. The unextracted audio is in residual.wav "
            f"({residual_db:.1f} dBFS RMS)."
        )

    asked_for_speech = remove_mode or any(
        w in (description or "").lower() for w in _SPEECH_WORDS)
    if not asked_for_speech:
        return None

    target_speech = _speech_band_fraction(target_path)

    if remove_mode:
        removed_speech = _speech_band_fraction(residual_path)
        # Two conditions, not one. "Merely higher" fires whenever the removed
        # sound carries incidental mid-band energy; a bare threshold fires on
        # any genuinely speech-adjacent source.
        if (removed_speech >= MIN_SPEECH_BAND_FRACTION
                and removed_speech > target_speech):
            return (
                f"What was REMOVED carries more speech than what was kept: "
                f"{removed_speech * 100:.1f}% of its energy is in the "
                f"{SPEECH_BAND_HZ[0]:.0f}-{SPEECH_BAND_HZ[1]:.0f}Hz speech band, against "
                f"{target_speech * 100:.1f}% in the audio you are keeping. SAM may have "
                f"treated the voice as '{description}' and discarded it. Check "
                f"residual.wav, which holds the removed sound."
            )
        if target_speech >= MIN_SPEECH_BAND_FRACTION:
            return None
        return (
            f"The audio kept after removing '{description}' is loud "
            f"({target_db:.1f} dBFS RMS) but contains almost no speech: only "
            f"{target_speech * 100:.1f}% of its energy is in the "
            f"{SPEECH_BAND_HZ[0]:.0f}-{SPEECH_BAND_HZ[1]:.0f}Hz speech band. SAM may have "
            f"removed far more than the sound you named."
        )

    if target_speech >= MIN_SPEECH_BAND_FRACTION:
        return None

    residual_speech = _speech_band_fraction(residual_path)
    msg = (
        f"Target output is loud ({target_db:.1f} dBFS RMS) but contains almost no speech: "
        f"only {target_speech * 100:.1f}% of its energy is in the "
        f"{SPEECH_BAND_HZ[0]:.0f}-{SPEECH_BAND_HZ[1]:.0f}Hz speech band. The description "
        f"'{description}' asked for a voice, so this is very likely the WRONG stem - SAM "
        f"appears to have extracted noise or rumble instead."
    )
    if residual_speech > target_speech:
        msg += (
            f" residual.wav carries {residual_speech * 100:.1f}% speech-band energy, so the "
            f"voice is probably THERE, not in target.wav."
        )
    msg += (
        " Note SAM-Audio is source separation, not noise reduction: for a single speaker "
        "with broadband noise, a spectral-subtraction denoiser is the better tool."
    )
    return msg
```

**Note:** keep the trailing lines of the original function (the `msg +=` "Note SAM-Audio is source separation…" and `return msg`) exactly as they were — read lines 261-266 before editing and preserve them verbatim.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_removal_guard.py tests/test_speech_band_sanity.py -q`
Expected: PASS — both the new tests and the existing guard tests, which cover the `remove_mode=False` path.

- [ ] **Step 5: Commit**

```bash
git add worker/handlers/sam_audio_cleanup.py tests/test_removal_guard.py
git commit -m "feat: two-sided sanity guard for removal jobs"
```

---

### Task 4: `remove_mode` swap inside `handle()`

**Files:**
- Modify: `worker/handlers/sam_audio_cleanup.py:415-422` (signature), `:653-656` (swap + guard call), `:683-708` (metadata)
- Test: `tests/test_remove_mode_swap.py` (create)

**Interfaces:**
- Consumes: `_separation_sanity_warning(..., remove_mode=...)` from Task 3
- Produces: `handle(..., remove_mode: bool = False)`; `metadata["method"] == "remove"` and `metadata["removed"] == <prompt>` when set; reads `input_data["original_description"]` for the user's untouched wording

**Deviation from the spec, deliberate.** The spec's Component 6 asked this task to assert
that `target + residual` reconstructs the input. It does not, for two reasons. First,
reaching that code requires loading SAM and a GPU, so it is not a unit test. Second and
more important, reconstruction is a property of *SAM's output*, not of the swap —
exchanging two file paths cannot break a sum. The assertion that actually pins this task
is **orientation** (which stem becomes `target.wav`) and **ordering** (that the swap
precedes every asymmetric consumer), which is what the test below checks by reading the
source. Reconstruction on real audio stays part of the un-done real-audio validation
recorded in Final Verification.

- [ ] **Step 1: Write the failing test**

Create `tests/test_remove_mode_swap.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_remove_mode_swap.py -q`
Expected: FAIL — `AssertionError: 'remove_mode' not found in ...` and `ValueError: substring not found` for the swap line.

- [ ] **Step 3: Write minimal implementation**

**3a.** Add the parameter to `handle()` (line 415-422), after `work_dir`:

```python
def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
    work_dir: Optional[Path] = None,
    remove_mode: bool = False,
) -> dict:
```

**3b.** Insert the swap and pass the flag to the guard (replacing lines 653-656):

```python
        target_path = _find_single_output(output_dir, "_target")
        residual_path = _find_single_output(output_dir, "_residual")

        # On a removal job the user wants everything EXCEPT what they named, and
        # that is exactly SAM's residual. Swapping here -- not after the ZIP is
        # built -- is what keeps the guard, both normalizers, sf.info and the ZIP
        # member names all pointing at the deliverable.
        if remove_mode:
            target_path, residual_path = residual_path, target_path

        separation_warning = _separation_sanity_warning(
            target_path, residual_path, description, remove_mode=remove_mode)
```

**3c.** Record the contract in metadata. After the `metadata = {...}` literal ends (line 706) and before the `if separation_warning:` block at line 707:

```python
        if remove_mode:
            metadata["method"] = "remove"
            metadata["removed"] = description
            # description here is the DERIVED prompt; surface the user's own
            # wording so a listener can see both what was asked and what was done.
            metadata["description"] = str(
                payload.get("original_description") or description)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_remove_mode_swap.py tests/test_speech_band_sanity.py tests/test_temp_cleanup.py -q`
Expected: PASS — including the existing handler tests, which exercise `remove_mode=False`.

- [ ] **Step 5: Commit**

```bash
git add worker/handlers/sam_audio_cleanup.py tests/test_remove_mode_swap.py
git commit -m "feat: swap stems inside handle() for removal jobs"
```

---

### Task 5: Wire the dispatch in `clean_cli.py`

**Files:**
- Modify: `clean_cli.py:224-256` (routing emit + dispatch block)
- Test: `tests/test_clean_cli_routing.py` (append a class)

**Interfaces:**
- Consumes: `strip_removal_cue()` (Task 1), `choose_method()` (Task 2), `handle(..., remove_mode=)` (Task 4)
- Produces: no new public names; `method == "remove"` dispatches to `handle()` with `remove_mode=True` and a payload whose `description` is the derived prompt

- [ ] **Step 1: Write the failing test**

Append to `tests/test_clean_cli_routing.py`:

```python
class RemoveDispatchTests(unittest.TestCase):
    """An unusable prompt must veto the route rather than reach SAM.

    handle() defaults an empty description to "speech" -- on a removal job that
    would extract exactly the thing the user wants to keep, then hand back the
    residual, deleting the voice. The veto is what prevents that.
    """

    def test_empty_prompt_downgrades_to_denoise(self):
        # "remove the" has a cue and no source, so routing already says denoise;
        # the veto covers the case where a source word survives routing but the
        # prompt strips to nothing.
        self.assertEqual(clean_cli.strip_removal_cue("remove the"), "")
        self.assertEqual(clean_cli.choose_method("remove the"), "denoise")

    def test_explicit_remove_with_unusable_prompt_is_vetoed_too(self):
        """An explicit method key does not make a blank prompt usable."""
        self.assertEqual(clean_cli.choose_method("remove", "remove"), "remove")
        self.assertEqual(clean_cli.strip_removal_cue("remove"), "")
```

- [ ] **Step 2: Run test to verify it passes already for routing, then implement the veto**

Run: `.venv/bin/python -m pytest tests/test_clean_cli_routing.py::RemoveDispatchTests -q`
Expected: PASS (these assert Task 1 + Task 2 behaviour). The veto itself is exercised through the dispatch code below; it has no unit seam of its own because `main()` requires a real audio file.

- [ ] **Step 3: Write the implementation**

In `clean_cli.py`, replace lines 224-227 (the routing block ending with the `emit(5, "route", ...)` call):

```python
        method = choose_method(payload.get("description", ""),
                               payload.get("method", "auto"))
        strength = payload.get("strength", "normal")

        # A removal job needs a prompt SAM can act on. If nothing usable
        # survives the strip, fall back to denoise: handle() defaults an empty
        # description to "speech", which on a removal job would extract the
        # voice and then hand back everything else.
        sam_payload = payload
        if method == "remove":
            prompt = strip_removal_cue(payload.get("description", ""))
            if prompt:
                sam_payload = dict(payload)
                sam_payload["description"] = prompt
                sam_payload["original_description"] = payload.get("description", "")
            else:
                method = "denoise"

        emit(5, "route", f"method={method}")
```

Then in the dispatch `else` branch (originally lines 247-256), pass the payload and flag:

```python
        else:
            with gpu_lock(args.gpu_lock_timeout):
                result = handle(
                    input_path=input_path,
                    input_data=sam_payload,
                    job={"id": 0, "input_filename": input_path.name},
                    progress_cb=lambda pct, msg, stage=None: emit(pct, stage or "processing", msg),
                    is_cancelled_cb=cancel.is_set,
                    work_dir=out_dir,
                    remove_mode=(method == "remove"),
                )
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS. Baseline was 148 passed / 6 warnings / 73 subtests; the count rises with the new tests, but **the warning count must still be exactly 6**.

- [ ] **Step 5: Commit**

```bash
git add clean_cli.py tests/test_clean_cli_routing.py
git commit -m "feat: dispatch removal jobs to SAM with remove_mode"
```

---

### Task 6: Update the README contract

**Files:**
- Modify: `README.md:207-236` (the "Name the sound you want to KEEP" and "Known routing limitations" sections), plus the `method` key description around line 248

**Interfaces:**
- Consumes: final behaviour from Tasks 1-5
- Produces: no code

- [ ] **Step 1: Read the current sections**

Run: `sed -n '200,260p' README.md`

The section headed **"Name the sound you want to KEEP, not the one you want gone"** documents the old ruling and is now wrong: it says removal phrasing denoises and "the dog will still be there".

- [ ] **Step 2: Replace that section**

```markdown
### Removing a sound vs. extracting one

Both are supported, and the difference is what lands in `target.wav`.

- **Name a sound** — `"the guitar"` — and SAM extracts it: `target.wav` is the guitar.
- **Ask for a sound to go** — `"remove the guitar"` — and you get the opposite:
  `target.wav` is everything *except* the guitar, and `residual.wav` holds the guitar
  that was taken out.

Removal only reaches SAM when the sound you name is one it can extract. "Remove the
background noise" and "remove the hiss" still denoise, because hiss is not a source —
asking SAM to extract it returns a near-empty stem.

If SAM cannot find the sound you asked to remove, the result is approximately your
original audio: nothing was removed, but nothing was lost either.
```

- [ ] **Step 3: Update the `method` key description**

At the `job.json` keys section, change the `method` line to:

```markdown
- `method` — `"auto"` (default), `"denoise"`, `"separate"`, or `"remove"`. An explicit
  value always beats inference. `"remove"` runs SAM on the named sound and returns the
  residual as `target.wav`.
```

- [ ] **Step 4: Fix the stale limitation bullet**

In "Known routing limitations", the mixture-word bullet says a bare "barking outside" "will denoise". That is still true, so leave it. Add one bullet:

```markdown
- **A removal request whose subject is the topic of the speech misroutes.** "Remove the
  part about traffic" names a subject being discussed, not a sound, and will route to
  `remove`. The two-sided guard should catch the resulting voice loss and warn, but set
  `method` explicitly when you know.
```

- [ ] **Step 5: Verify and commit**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, 6 warnings.

```bash
git add README.md
git commit -m "docs: removal returns the residual, not the denoiser"
```

---

## Final Verification

- [ ] Full suite green: `.venv/bin/python -m pytest tests/ -q` — **exactly 6 warnings**
- [ ] `git status --short` shows only the 3 permitted untracked files
- [ ] The 42-description must-denoise corpus in `tests/test_clean_cli_routing.py` passes **unmodified**
- [ ] **Real-audio validation is NOT done and must not be claimed.** `Fionnuala_raw.wav` is single-speaker with nothing to remove, so it cannot exercise this route. The swap is verified synthetically only. Ask Paul for a mixture file containing a removable source (music, a dog, traffic) before reporting this as end-to-end verified.
