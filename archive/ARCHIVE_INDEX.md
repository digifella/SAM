# Archive Index

This directory contains archived versions of scripts and documentation from the SAM-Audio project development process.

## Directory Structure

### old_scripts/
Previous versions of the SAM-Audio processing scripts. These have been superseded by `run_sam_interactive.py`.

**Archived scripts:**
- `run_sam_audio.py` - Original single-file processor
- `run_sam_audio_batch.py` - First batch processing attempt
- `run_sam_audio_batch_chunked.py` - Early chunking implementation
- `run_sam_audio_batch_robust.py` - Robustness improvements
- `run_sam_audio_patched.py` - Memory patch attempt
- `run_sam_audio_simple_batch.py` - Simplified batch processor
- `run_sam_audio_ultrarobust.py` - Error handling improvements

### screenshots/
Screenshots documenting crash issues during development:
- `Screenshot 2026-01-12 104121.jpg` - Initial crash evidence
- `Screenshot 2026-01-12 111745.jpg` - Memory exhaustion crash

### old_docs/
Previous documentation and troubleshooting guides:
- `AGENTS.md` - Agent configuration documentation
- `README_ROBUSTNESS.md` - Robustness implementation notes
- `TROUBLESHOOTING.md` - Early troubleshooting guide
- `UPGRADE_QUICK_START.md` - Upgrade instructions
- `WHICH_SCRIPT_TO_USE.md` - Script comparison guide
- `WSL2_CRASH_DIAGNOSIS.md` - WSL2 crash analysis
- `WSL2_CRASH_RESOLVED.md` - Initial crash resolution
- `WSL2_UPGRADE_PLAN.md` - Upgrade planning document

## Current Production Version

**Active script:** `run_sam_interactive.py`

**Current documentation:** `CRASH_FIX_2026-01-12.md`

This version implements:
- Memory-safe streaming chunk processing
- Generator-based audio chunking to prevent memory exhaustion
- Aggressive memory cleanup between chunks
- WSL2 stability improvements

## History

All archived files represent the development journey to solve memory exhaustion crashes when processing long audio files on WSL2 systems.

**Final solution (Jan 12, 2026):**
- Converted chunk loading from batch to streaming generator
- Fixed infinite loop in chunk iteration
- Successfully processes 300-second files without crashing
- Memory usage remains stable at ~12.5 GB throughout processing
