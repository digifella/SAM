# SAM-Audio: Quality Review, Cortex Integration, Hermes Discord Loop — Design

**Date:** 2026-08-01
**Status:** Approved by Paul (this session)
**Repos touched:** `sam-audio` (primary), `cortex_suite` (one page), fastfella host scripts (`kb-query-server.py`), sp4 (`~/.local/bin/kb-audio-clean` + Hermes tool registration)

## Goal

Three workstreams, implemented incrementally so a reboot/BIOS update can land between any two phases:

1. **Quality/bug/update review** of sam-audio, plus a verified capacity gate: a ~45-minute voice recording (speech with background noise / poor mic) processes end-to-end on the RTX 8000 without OOM.
2. **Cortex Suite integration**: clean up audio from a new page inside the existing Cortex Streamlit app.
3. **Hermes Discord loop**: post audio in Discord → Hermes (on sp4) submits it to fastfella → cleaned audio is posted back to the Discord channel automatically.

## Decisions made (with Paul, 2026-08-01)

- Hermes already downloads Discord attachments to a local path on sp4 — a tool wrapper can be pointed at that file.
- **Return path: Discord webhook push from fastfella.** Hermes never blocks on the 10–30 min job; it replies "submitted" and the finished audio is posted by fastfella directly to the channel webhook.
- **Return format: OGG/Opus ~48kbps mono** (~16MB for 45 min — fits Discord's 25MB cap; excellent for speech). Full-quality WAV ZIP stays on fastfella in the job's spool directory (kept 7 days), retrievable from the filesystem when needed.
- **Cortex page UX: simplified preset** (upload → description → Clean), advanced knobs in an expander.
- **Architecture: shared CLI + job spool** (Approach A). No new always-on daemon; VRAM is released between jobs; the existing token-gated kb bridge is extended rather than adding a new service.

Rejected: persistent FastAPI model-warm service (permanent 13GB VRAM hold, one more daemon to babysit); routing via the website cPanel queue (unusable for 100–500MB uploads).

## Architecture

```
                 ┌──────────────── fastfella (this WSL host) ────────────────┐
Discord user     │                                                           │
   │ audio       │  kb-query-server.py (Tailscale 100.118.92.17:7333)        │
   ▼             │    POST /audio-clean ──► spool/<job_id>/ ──► spawn        │
SlowClaw/Hermes ─┼──►                                            detached    │
  (sp4)          │    GET  /audio-status?id=◄── status.json      │           │
  kb-audio-clean │                                               ▼           │
                 │  sam-audio/.venv/bin/python clean_cli.py                  │
                 │    └─ worker/handlers/sam_audio_cleanup.handle (existing) │
                 │    └─ flock GPU lock … result.zip + status.json           │
                 │    └─ --opus → target.ogg                                 │
                 │    └─ --discord-webhook → POST file+summary to channel    │
                 │                                                           │
                 │  cortex_suite pages/21_Audio_Cleanup.py                   │
                 │    └─ subprocess clean_cli.py (same CLI, JSON-line        │
                 │       progress → st.progress), play/download results      │
                 └───────────────────────────────────────────────────────────┘
```

## Components

### 1. `clean_cli.py` (new, sam-audio repo root)

Thin CLI over the existing `worker/handlers/sam_audio_cleanup.handle`:

- Args: `--input <file>`, `--job-json <params.json>` (same payload schema the handler already accepts), `--out-dir <dir>`, `--opus`, `--discord-webhook <url>` (or env `DISCORD_WEBHOOK_URL`).
- Emits progress as JSON lines on stdout: `{"type":"progress","pct":35,"stage":"separate","message":"…"}`.
- Writes `<out-dir>/result.zip` (target.wav, residual.wav, metadata.json — as today) and a final `<out-dir>/status.json` (`{"state":"done"|"error", "error":…, "metadata":…, "started_at":…, "finished_at":…}`).
- `--opus`: after success, `ffmpeg -i target.wav -ac 1 -c:a libopus -b:a 48k target.ogg` into out-dir.
- `--discord-webhook`: on success POST multipart (target.ogg + summary text: input name, duration, description used, any near-silent warning); on failure POST the error message. Webhook post failures are logged but don't fail the job.
- GPU serialization: `flock` on `~/.sam_audio_gpu.lock` around model load + separation, with a long timeout and a "waiting for GPU lock" progress event.
- Runs only under `sam-audio/.venv` (Python 3.11, torch already installed).

### 2. Cortex page `pages/21_Audio_Cleanup.py` (cortex_suite repo)

- Simplified UX: file upload (wav/mp3/flac/ogg/m4a/aac/mp4/mkv/mov) → "What to extract" text box (default `speech`) → **Clean Audio** button → progress bar + status line → inline players for target/residual → download ZIP button.
- Advanced expander: chunk duration, overlap, loudness normalize checkbox, trial-first-N-seconds.
- Invocation: `subprocess.Popen(["/home/longboardfella/sam-audio/.venv/bin/python", ".../clean_cli.py", …])`, reading JSON-line progress from stdout. Path configurable via `SAM_AUDIO_ROOT` env, default `/home/longboardfella/sam-audio`.
- No torch/SAM dependencies added to the cortex venv. Follow cortex page conventions and run `python scripts/version_manager.py --sync-all` per cortex CLAUDE.md.

### 3. Bridge endpoints (fastfella `kb-query-server.py`)

- `POST /audio-clean` — token-gated like every other endpoint. Multipart upload: `file` + optional fields `description` (default `speech`), `loudness` (bool). Writes to `~/sam-audio/spool/<job_id>/input.<ext>` (job_id = timestamp+random). Spawns `clean_cli.py` **detached** (survives bridge restart) with `--opus --discord-webhook`. Returns `{"job_id": …}` immediately. Upload cap ~2GB.
- `GET /audio-status?id=<job_id>` — returns the job's `status.json` (or `{"state":"running"}` if absent, `{"state":"stale"}` if the process is gone and no status was written — e.g. killed by a reboot).
- Spool dirs older than 7 days are purged on bridge start.
- `DISCORD_WEBHOOK_URL` read from the bridge's env file on fastfella — never committed to git.

### 4. sp4 side

- `/home/paul/.local/bin/kb-audio-clean <file> [description…]`: curl -F upload to `http://100.118.92.17:7333/audio-clean`, prints the job ID + "cleaned audio will be posted to Discord when done". Matches existing kb-* wrapper conventions (token header etc. — copy from an existing wrapper).
- Hermes tool registration/prompt nudge so SlowClaw knows: audio attachment + a cleanup request → call `kb-audio-clean` with the attachment's local path → reply that the job is submitted.
- **One manual step for Paul:** create the Discord webhook in the target channel (channel settings → Integrations → Webhooks) and give me the URL for the fastfella env file.

## Workstream 1: review scope + capacity gate

Review targets (fix + commit each finding separately):

- `worker/handlers/sam_audio_cleanup.py` — **known bug:** creates `tempfile.mkdtemp` work_dir that is never removed on the Streamlit path (worker.py only cleans its own dir; the comment at the end of `handle` is wrong). Leaks a result ZIP per job into /tmp.
- `streamlit_app.py` — **known bug:** `st.experimental_rerun()` is removed in current Streamlit; check the installed version and use `st.rerun()`.
- `run_sam_interactive.py` (1508 lines), `worker/worker.py`, `sam_audio_local/loader.py` — correctness pass (error paths, cancellation, cleanup, dtype boundaries).
- Dependency currency: torch / streamlit / soundfile pins vs. current; note but don't chase upgrades that risk the working fp16 build.

Capacity gate (success criteria for "45 min works"):

- A real ~45-minute voice-with-background file processes end-to-end on GPU: no OOM-ladder engagement, flat memory across chunks, runtime + peak VRAM + peak WSL RAM recorded in the plan doc.
- Output spot-checked audibly (target = clean voice, residual = background).

## Error handling

- Separation errors: existing OOM retry ladder + CPU fallback in the handler (unchanged).
- Discord jobs: any failure POSTs an error message to the webhook so it's visible in-channel; status.json carries the error for `audio-status`.
- Reboot mid-job: spool persists; job shows `stale`; resubmit from Discord. No auto-resume (deliberate YAGNI).
- Bridge: 4xx on missing/oversized upload; 403 on bad token (existing mechanism).

## Testing

- Unit (sam-audio pytest, extend existing suite): CLI arg parsing + JSON-line protocol + status.json writing; opus encode (tiny fixture wav); webhook post with mocked HTTP; handler temp-dir cleanup regression test.
- Bridge: endpoint tests with a stub CLI (no GPU) — upload → spool layout → detached spawn → status readback.
- Integration: short clip via curl → bridge → real webhook into a test channel; then full Discord round-trip from sp4; then the 45-min soak; manual Cortex page run.

## Implementation phasing (interruption-safe)

Each phase is independently verifiable and committed; a reboot between phases loses nothing. Detailed steps go in the implementation plan doc (next step: writing-plans skill).

- **Phase 0:** review findings fixed + committed (sam-audio)
- **Phase 1:** 45-min capacity soak, results recorded
- **Phase 2:** `clean_cli.py` + tests (foundation)
- **Phase 3:** Cortex page
- **Phase 4:** bridge endpoints + opus + webhook (needs webhook URL from Paul)
- **Phase 5:** sp4 wrapper + Hermes registration
- **Phase 6:** end-to-end Discord round-trip test
