# Dependency Currency Review — 2026-08-02

**Purpose:** record dependency drift for `sam-audio`'s `.venv`, and the decision
not to chase it. This is a documentation snapshot, not an upgrade action — no
package was installed, uninstalled, or upgraded to produce this file.

Environment: `.venv` = `/home/longboardfella/venvs/sam-audio311` (Python 3.11,
pip 26.0.1), repo `sam-audio` on branch `feat/quality-cortex-hermes` at
commit `a90709b`.

## Method

Two read-only pip commands were run from the `sam-audio` repo root:

```bash
.venv/bin/pip list --format=columns
.venv/bin/pip list --outdated --format=columns
```

The `--outdated` check queries PyPI once per installed package (241 packages
in this venv) and took several minutes to complete — plain shell timeouts of
60–90s were insufficient and returned no output; run to completion in the
background it finished with exit code 0 and full results. Network reachability
to `pypi.org` was confirmed separately (`curl -sI https://pypi.org/simple/` →
`HTTP/2 200`), so the earlier timeouts were a duration issue, not a
connectivity failure. No `--outdated` data had to be omitted.

## Current pins — as observed, verified against the brief

The brief named five pins as the current state. Each was checked against real
`pip list --format=columns` output rather than transcribed from the brief.
**All five match exactly — no discrepancy found:**

| Package    | Brief claimed | Observed (`pip list`) | Match |
|------------|---------------|------------------------|-------|
| torch      | 2.6.0+cu124   | 2.6.0+cu124            | yes   |
| streamlit  | 1.19.0        | 1.19.0                 | yes   |
| soundfile  | 0.13.1        | 0.13.1                 | yes   |
| numpy      | 1.26.4        | 1.26.4                 | yes   |
| requests   | 2.32.5        | 2.32.5                 | yes   |

Related torch-ecosystem packages observed alongside the `torch` pin (same
`pip list` run, relevant because any future torch bump would need to move
these in lockstep):

```
torch                     2.6.0+cu124
torch-stoi                0.2.3
torchaudio                2.6.0+cu124
TorchCodec                0.2.1+cu124
torchdata                 0.11.0
torchdiffeq               0.2.5
torchlibrosa              0.1.0
torchvision               0.21.0
```

## What is outdated (`pip list --outdated`, real output)

150 of 241 installed packages report a newer version on PyPI — typical for an
ML stack that has been intentionally left alone since the fp16/RTX-8000 build
was validated. The full 150-line output is not reproduced here; the rows for
the five reviewed pins plus directly-coupled packages are:

```
Package                  Version            Latest             Type
------------------------ ------------------ ------------------ -----
altair                   4.2.2              6.2.2              wheel
numpy                    1.26.4             2.4.6               wheel
protobuf                 3.19.6             7.35.1              wheel
requests                 2.32.5             2.34.2              wheel
soundfile                0.13.1             0.14.0              wheel
streamlit                1.19.0             1.60.0              wheel
torch                    2.6.0+cu124        2.13.0               wheel
torchaudio               2.6.0+cu124        2.11.0               wheel
TorchCodec               0.2.1+cu124        0.15.0               wheel
torchvision              0.21.0             0.28.0               wheel
```

`numpy` is two major versions behind (1.26 → 2.4), `torch` is roughly seven
minor releases behind (2.6 → 2.13), and `streamlit` is over a year of releases
behind (1.19 → 1.60).

`streamlit_requirements.txt` in this repo pins `streamlit==1.19.0`,
`altair<5`, and `protobuf<3.20` together, with the comment: *"Keep Streamlit
on a protobuf<3.20-compatible release because sam_audio ->
descript-audiotools==0.7.2 requires protobuf<3.20."* This confirms the
streamlit pin is not just a Turing-caution choice — it is also load-bearing
for the `descript-audiotools` dependency chain via `protobuf`. Any future
streamlit bump would need `protobuf` and `altair` to move together and would
need `descript-audiotools`'s protobuf constraint re-checked at the same time.

## Decision: hold torch and streamlit pins

**No torch or streamlit upgrade is being made as part of this review, and
none is recommended without a separate, explicit verification pass.**

Reasoning:

- The current stack (torch 2.6.0+cu124, streamlit 1.19.0) is a **working fp16
  build validated on a Quadro RTX 8000 (Turing, SM 7.5)**. This is documented
  in `README.md` ("Memory-Optimized Local Loading (RTX 8000 / 48GB)" — casting
  the DiT/codec to fp16, "native on Turing").
- Turing (SM 7.5) support in newer torch wheels is not guaranteed and has not
  been re-verified against 2.7+. Newer PyTorch releases have progressively
  narrowed default CUDA-architecture coverage in prebuilt wheels; whether
  SM 7.5 kernels are still shipped/built for a given newer torch+cu12x wheel
  needs to be checked release-by-release before any bump — this review does
  not do that check.
- `streamlit` is additionally pinned by the `protobuf<3.20` requirement
  described above; moving it requires resolving that constraint too.
- This task's scope is explicitly "record, don't chase" — no upgrade,
  install, or requirements-file edit was performed.

If a future maintainer wants to move off these pins, the work is: (1) confirm
a target torch/cu12x wheel still builds/ships SM 7.5 kernels, (2) test the
fp16 DiT/codec path end-to-end on the RTX 8000, (3) separately resolve the
streamlit/protobuf/altair/descript-audiotools constraint chain before bumping
streamlit off 1.19.0.

## Security-relevant advisories — requests 2.32.5, streamlit 1.19.0

Per the task instructions, only what is actually known with confidence is
stated below; everything else is explicitly marked unverified rather than
guessed. **No CVE identifiers are asserted in this document — none were
confirmed against a CVE database during this review.**

- **requests 2.32.5**: this is a comparatively recent release. I am not aware
  of any specific, confirmed CVE affecting 2.32.5 itself. `requests` has had
  past security advisories in earlier release lines (historically around
  `.netrc` credential handling and proxy-header/redirect behavior), but I do
  not have verified confirmation of exact CVE IDs, affected version ranges, or
  fix versions at the level of confidence needed to cite them here — **treat
  this as unverified — needs check against a CVE database** (e.g. the GitHub
  Advisory Database or OSV.dev) before relying on this observation for any
  security decision.
- **streamlit 1.19.0**: this release is from 2023 and is significantly behind
  current (1.60.0 observed above). I do not have specific, confirmed
  knowledge of a CVE affecting 1.19.0 — **unverified — needs check against a
  CVE database.** Given the version gap (40+ releases), an actual advisory
  check is recommended before this pin is trusted for anything
  network/auth-adjacent, independent of the "hold" decision above (which is
  driven by hardware/build-stability concerns, not a security assessment).

**Action for a future maintainer:** before relying on this document for a
security posture claim, run an actual advisory scan (e.g. `pip-audit`,
`safety`, or a manual OSV/GHSA lookup) against the exact pinned versions.
This review did not do that — it is a currency/decision record, not a
security audit.
