# 45-Minute Capacity Soak — Results (RTX 8000)

**Task 5 of** `docs/superpowers/plans/2026-08-01-quality-cortex-hermes-integration.md`.
Run 2026-08-02 21:38 → 21:55 AEST on `feat/quality-cortex-hermes`.

**Verdict: GATE PASSED** — criteria 1, 2, 3 and 5 pass on measurement; criterion 4
(peak RAM) was **waived by the user** after re-measurement (see below). No criterion
was silently relaxed.

## Input

`soak_45min.wav` = `Apollo13.wav` (57.6s of noisy radio voice) looped 46×
via `ffmpeg -stream_loop 46 -c copy`. Measured duration **2708.891s (45.15 min)**,
matching the plan's expected ~2708s.

Job parameters: `description="a man speaking over a radio"`, `convert_to_mono=True`,
`chunk_duration=60`, `overlap=2.0` — i.e. the default 60/2 profile.

## Headline numbers

| Metric | Value |
|---|---|
| Elapsed | 1022.8 s |
| Audio processed | 2708.891 s |
| **Realtime factor** | **2.65× faster than realtime** |
| Chunks | 47 / 47 |
| Output sample rate | 48000 Hz, mono |
| Peak VRAM | 32933 MiB (of 46080) |
| Peak host RAM | 21664 MiB (of 24033) |
| Exit | 0 |

## Gate criteria

| # | Criterion | Result |
|---|---|---|
| 1 | Job completes | **PASS** — exit 0, zip written, 47/47 chunks |
| 2 | `auto_profile == "requested"` | **PASS** — OOM ladder never engaged; ran attempt 1/3 throughout |
| 3 | Peak VRAM < 44000 MiB | **PASS** — 32933 MiB at t+534s (11067 MiB headroom) |
| 4 | Peak used RAM < 20000 MiB | **FAIL as written → WAIVED by user** — 21664 MiB at t+92s |
| 5 | Memory flat across chunks | **PASS** — decisively, see below |

### Criterion 5 — flatness

Measured over the 89 monitor samples where VRAM > 10GB (t+123s … t+1029s, i.e. real
chunk processing, excluding model load and teardown):

```
VRAM  min/mean/max  19889 / 26735 / 32933 MiB
      thirds mean   26358 -> 27160 -> 26674   (drift +316 MiB = noise)
RAM   min/mean/max   6687 /  6755 /  7063 MiB
      thirds mean    6797 ->  6734 ->  6736   (drift -61 MiB)
```

Host RAM spans just **376 MiB across 15 minutes** of processing. No monotonic climb,
no leak. This is the criterion that actually answers "can it do 45 minutes", and it
passes without qualification.

### Criterion 4 — the RAM peak, and why it is waived

The 21664 MiB peak is a **model-load transient, not a capacity-vs-duration problem.**

Only 2 of 164 soak samples ever reached 20000 MiB (t+92s and t+103s), both while VRAM
was still ~7093 MiB — i.e. the SAM weights being materialised in host RAM before
transfer to the GPU. By t+123s RAM was back to ~6.7GB and stayed there.

Because the concurrent Batch A review and Batch B implementer were dispatched at
~21:39-21:40 (around t+92s), contamination could not initially be ruled out. It was
therefore **re-measured on a quiet machine** (`ram_probe.py`, 0.5s sampling vs the
soak monitor's 10s, nothing else running, RAM idle 2926 MiB):

```
PEAK RAM 21708 MiB at t+55.7s, VRAM only 7699 MiB
34 samples >= 20000 MiB spanning t+39.9 .. t+58.5  = a 17-SECOND window
Peak VRAM in probe 32885 MiB (matches the soak's 32933)
```

The probe peak is *higher* than the soak's, so the concurrent agents did not cause it.
Critically, the probe ran on **Apollo13.wav — 57.6 seconds** — so the peak is
**duration-independent**: it is a fixed model-load cost that a 1-minute file incurs
identically. The plan framed criterion 4 as part of a *45-minute capacity* gate, but as
written no input of any length can meet it on this host.

**User ruling (2026-08-03):** *"you can use RAM as required unless wsl is limiting you.
I am not arbitrarily limiting it."* → criterion 4 waived; the 21708 MiB peak is recorded
here as a documented characteristic, not a failure.

Facts established so the ruling is applied with eyes open — **WSL is limiting, deliberately:**

- `/mnt/c/Users/paul/.wslconfig` sets `memory=24GB` of 63.8GB physical, with the in-file
  comment *"leaves 40GB for Windows"*; `swap=16GB`. The 24033 MiB ceiling is a config choice.
- That 40GB is not spare: LM Studio (~32GB with `qwen3.6-35b-a3b` resident) and ComfyUI
  Desktop both run host-side. Raising the WSL cap takes RAM directly from them.
- `.wslconfig` also carries `autoMemoryReclaim=gradual`, commented with a 2026-06-05
  near-freeze caused by vmmem ballooning — prior art for host RAM pressure hurting this box.
- Swap is 16GB (2844 MiB in use at measurement), so a spike degrades to swap rather than
  OOM-killing.

**Recommendation: do NOT raise the WSL cap.** The 21708 MiB peak already fits (the 45-min
soak completed cleanly) and raising it would rob the Windows-side inference stack.
**No action taken.**

**Residual operational risk (carry into Tasks 6/10/11):** the peak leaves only ~2.3GB
headroom for a ~17s window on *every* job. A concurrent WSL-side RAM consumer during that
window could trigger an OOM kill. The nightly 1am vault reindex is a known RAM-heavy job on
this box — **audio jobs should not overlap it.**

## Output quality

Guards against a "passed but produced garbage" soak:

- `soak_run.log` contains **no** near-silent warning.
- Target RMS measured at three points: start **-25.2**, middle **-23.0**, end **-26.0** dBFS,
  against input Apollo13 at -20.7 dBFS. Healthy levels, and consistent across all three
  points — **no degradation over the 45 minutes**.

Preserved artefacts (`/tmp` is reboot-cleared, these are not):

- `audio_output/soak_result.zip` (319MB — target.wav, residual.wav, metadata.json; gitignored)
- `audio_output/soak_check_{start,middle,end}.wav` — 25s clips at 0:30, 22:20, 44:10 for the
  Step 6 user listening gate.

> **Step 6 (user listening gate) remains OUTSTANDING** — Paul to confirm the voice is clean
> and the background landed in the residual.

## Task 3 outcome note

Task 3 (correctness review scan) returned **8 findings — all 8 confirmed** on triage by
reopening the cited lines, and all fixed across three batches:

- **Batch A** (`run_sam_interactive.py`): F1 merge_chunks buffer formula (Critical),
  F7 mkstemp leak, F8 orphaned mkdtemp on ffmpeg failure.
- **Batch B** (`worker.py` + cancel sites): F3 unprotected `complete()` (Critical, data loss),
  F4 mkdtemp/heartbeat outside try (Important), F2a cancel sites raising plain `RuntimeError`.
- **Batch C** (`sam_audio_cleanup.py`): F5 cap-limited OOM detection never firing (Important),
  F6 retained traceback pinning GPU tensors across the retry ladder (Important).

**F2b escalated and NOT built:** no cancellation *source* exists — `client.heartbeat()`
discards its response and `QueueClient` has no cancel endpoint. Wiring real cancellation is a
queue-server protocol change outside this plan's four in-scope files. **Awaiting user decision.**

Suite grew 42 → 69 tests across the plan; full suite green at time of writing
(69 passed, 6 known pre-existing third-party warnings).

## Caveats on this measurement

- The soak monitor sampled every 10s, so the true peak may exceed any sampled row. The 0.5s
  `ram_probe` was run specifically to address this for the RAM figure.
- The soak ran concurrently with Task 3 Batch A's review and Batch B's implementer (a
  deliberate sequencing decision — the soak calls `handle()` directly, imports its code at
  process start, and its gate requires the OOM ladder never engage). The RAM figure was
  re-measured alone; VRAM and timing were not, but both passed with wide margins.

## Short files and the non-chunked path — CONFIRMED DEFECT

> **⚠️ RETRACTION WITHDRAWN (2026-08-03, later the same day).** An earlier version of this
> section retracted the defect as "not reproducible". **That retraction was wrong** and is
> itself now withdrawn. The defect is real; it is *intermittent*, which is why one clean run
> appeared to refute it. The original text is kept below the evidence table for the record.
>
> **The non-chunked path fails roughly two runs in three. The chunked path has never failed.**
>
> Apollo13.wav, 57.64s, input −20.7 dBFS, description "a man speaking over a radio",
> every run identical except `chunk_duration`. Target RMS in dBFS; below −50 is silence:
>
> | Run | Path | Target | Verdict |
> |---|---|---|---|
> | `ram_probe` | non-chunked | −70.9 | **SILENT** |
> | controlled experiment A | non-chunked | −18.1 | healthy |
> | Task 9 CLI smoke | non-chunked | −78.1 | **SILENT** |
> | flakiness rep 1 | non-chunked | −71.0 | **SILENT** |
> | flakiness rep 2 | non-chunked | −70.6 | **SILENT** |
> | flakiness rep 3 | non-chunked | −24.0 | healthy (weak) |
> | controlled experiment B | chunked (cd=30) | −17.8 | healthy |
> | flakiness rep 1 | chunked (cd=30) | −18.1 | healthy |
> | flakiness rep 2 | chunked (cd=30) | −19.5 | healthy |
> | flakiness rep 3 | chunked (cd=30) | −19.6 | healthy |
>
> **non-chunked: 4 silent / 6. chunked: 0 silent / 4**, and the chunked results cluster
> tightly (−17.8 to −19.6) while even the non-chunked "healthy" outliers are weaker.
>
> **Method error worth naming, because it is the reusable lesson:** the retraction ran
> *one* trial per condition, got a healthy non-chunked result, and concluded the defect did
> not exist. A single passing sample cannot refute an intermittent failure — it can only fail
> to reproduce it. "Not reproduced in this run" is not "not a defect". Flaky behaviour needs
> repetition sized to the claim before either confirming or dismissing it.
>
> **Consequence:** this matters for `clean_cli.py` (Task 6) and the Cortex page (Task 10),
> which routinely send sub-60s files — and for the Discord loop, where voice notes are
> typically 10–30s and would land on the non-chunked path every time.

### Original (withdrawn) retraction text, kept for the record

The original claim: the `ram_probe` run took the non-chunked path (57.64s < 60s
`chunk_duration` → "Processing entire file") and produced a near-silent target at
**-70.9 dBFS** RMS, implying SAM extracted nothing on short files — which would have
affected Task 6 (`clean_cli`) and Task 10 (the Cortex page), since both routinely send
sub-60s files down that path.

**A controlled experiment on 2026-08-03 does not reproduce it.** Same file
(`Apollo13.wav`, 57.64s), same description, same `convert_to_mono`/`overlap`; the *only*
variable was `chunk_duration`, chosen to force each branch:

| Condition | `chunk_duration` | Branch taken (from log) | Target RMS | Residual RMS | Elapsed |
|---|---|---|---|---|---|
| A | 60 | `Processing entire file (no chunking needed)` | **-18.1 dBFS** | -23.9 dBFS | 117.4s |
| B | 30 | `Using chunking (chunk size: 30.0s, overlap: 2.0s)`, 3 chunks | **-17.8 dBFS** | -21.2 dBFS | 25.5s |

Both outputs are healthy and within 0.3 dB of each other, against an input at -20.7 dBFS.
The non-chunked path separates correctly. The final 1.6s chunk in condition B (56.0–57.6s)
also processed cleanly, so short trailing chunks are fine too.

**Conclusion: there is no short-file defect, and no chunking-fallback is needed.** The
-70.9 dBFS figure came from `ram_probe.py`, a throwaway script written for the RAM
measurement and since lost with `/tmp`. Its parameters could not be recovered, so the
confounding variable cannot be named with certainty — but it was *not* the chunked/
non-chunked distinction, which is the only thing the original claim attributed it to.

**Lesson:** the probe was built to measure RAM, and its audio output was judged
opportunistically as a side observation. A measurement rig is not a correctness rig —
a side observation from one should be reproduced under control before it is recorded
as a defect.
