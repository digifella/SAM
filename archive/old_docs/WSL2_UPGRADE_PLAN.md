# WSL2 Upgrade & Stability Testing Plan
**Date:** 2026-01-11
**Current WSL Version:** 2.5.7.0
**Target:** Latest WSL2 (2.7.x+)

---

## Pre-Upgrade Status

### Hardware Configuration
- **RAM:** 64GB total
  - Memory speed: **4600MHz** (reduced from higher speed)
  - XMP: **Disabled** in BIOS
  - Physical hardware: **Reseated** (RAM + NVIDIA card)
- **GPU:** Quadro RTX 8000 (48GB VRAM)
  - Driver: 582.16
  - Status: Working correctly
- **CPU:** 24 cores available

### Current WSL2 Configuration
- Memory allocated: 32GB to WSL2
- Swap: 16GB (active and working)
- Processors: 16 cores
- Config location: `C:\Users\paul\.wslconfig`

### Known Issues Fixed
1. ✅ Invalid config keys removed (autoMemoryReclaim, sparseVhd)
2. ✅ Swap file working correctly
3. ✅ GPU detected properly after reseating

---

## Upgrade Process

### Step 1: Backup Current Configuration
Already documented in:
- `/home/longboardfella/sam-audio/WSL2_CRASH_DIAGNOSIS.md`
- `/home/longboardfella/sam-audio/README_WSL_CRASH.txt`
- Current .wslconfig saved

### Step 2: Upgrade WSL2 (Run in PowerShell as Administrator)
```powershell
# Check current version
wsl --version

# Update to latest version
wsl --update

# Verify new version
wsl --version

# Shutdown WSL2
wsl --shutdown
```

**Expected new version:** 2.7.0 or higher

### Step 3: Update .wslconfig with New Features
After upgrade, add these lines back to `.wslconfig`:
```ini
autoMemoryReclaim=gradual  # Reclaim memory gradually
sparseVhd=true             # Allow virtual disk to shrink
```

Full updated config will be in: `C:\Users\paul\.wslconfig`

### Step 4: Restart WSL2
```powershell
wsl --shutdown
# Wait 10 seconds
# Restart WSL terminal
```

---

## Post-Upgrade Verification (Run After WSL Restart)

### Quick Check Script
Run this immediately after restart:
```bash
bash /home/longboardfella/sam-audio/post_upgrade_verify.sh
```

### Manual Verification
```bash
# 1. Check WSL version
wsl.exe --version

# 2. Check memory and swap
free -h
swapon --show

# 3. Check GPU
nvidia-smi

# 4. Check for errors
dmesg | tail -50 | grep -E "error|fail|oom" -i

# 5. Verify Python environment
source /home/longboardfella/sam-audio/.venv/bin/activate
python --version
pip list | grep torch
```

---

## Stability Testing Plan

### Test 1: Memory Stress Test (5 minutes)
```bash
bash /home/longboardfella/sam-audio/stability_test.sh
```

This will run:
1. System RAM stress test (5 min)
2. GPU memory test
3. Combined CPU+RAM test
4. Memory leak detection

### Test 2: Real Workload Test
Run a small SAM-Audio job to verify stability:
```bash
# Monitor in separate terminal first
watch -n 1 'free -h && echo --- && nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader'

# Then run small test
python run_sam_audio_ultrarobust.py --file mnt/f/hf-home/audio_input/Campbell_Audio.wav --desc "test run" --verbose
```

---

## Expected Results

### After Upgrade
- WSL version: 2.7.x or higher
- No config errors when starting WSL
- Memory: 32GB allocated
- Swap: 16GB active
- GPU: Detected correctly
- Python environment: All packages intact

### After Stability Tests
- No crashes or freezes
- Memory stays within limits
- Swap used if needed (but shouldn't crash)
- GPU memory managed correctly
- System responsive throughout

---

## Rollback Plan (If Issues Occur)

### If WSL2 Upgrade Fails
```powershell
# Uninstall WSL2 update
wsl --unregister Ubuntu
wsl --install Ubuntu

# Restore from backup (if needed)
```

### If Stability Issues Persist

**Check these in order:**
1. Memory speed in BIOS (currently 4600MHz)
2. XMP disabled (currently disabled)
3. Test RAM modules individually
4. Check Event Viewer for hardware errors:
   ```powershell
   Get-WinEvent -LogName System -MaxEvents 100 | Where-Object {$_.LevelDisplayName -eq "Error"}
   ```

---

## Files Created

- **This file:** `/home/longboardfella/sam-audio/WSL2_UPGRADE_PLAN.md`
- **Verification script:** `/home/longboardfella/sam-audio/post_upgrade_verify.sh`
- **Stability test:** `/home/longboardfella/sam-audio/stability_test.sh`
- **Updated config:** Will be at `C:\Users\paul\.wslconfig`

---

## Commands to Run (In Order)

### On Windows (PowerShell as Administrator):
```powershell
# 1. Upgrade WSL2
wsl --update

# 2. Check version
wsl --version

# 3. Shutdown
wsl --shutdown
```

### After WSL Restart (Linux):
```bash
# 1. Navigate to project
cd /home/longboardfella/sam-audio

# 2. Verify installation
bash post_upgrade_verify.sh

# 3. Run stability tests
bash stability_test.sh

# 4. If all good, test SAM-Audio
source .venv/bin/activate
python run_sam_audio_ultrarobust.py --file mnt/f/hf-home/audio_input/Campbell_Audio.wav --desc "stability test" --verbose
```

---

## Current Session State

**Working directory:** `/home/longboardfella/sam-audio`
**Python venv:** `.venv` (needs to be reactivated after restart)
**GPU:** Quadro RTX 8000 - working
**Previous crash cause:** No swap - NOW FIXED
**Hardware changes:** RAM + GPU reseated, memory speed reduced to 4600MHz

---

## Success Criteria

- [ ] WSL2 upgraded to 2.7.x+
- [ ] No config errors on startup
- [ ] Swap working (16GB shown)
- [ ] GPU detected
- [ ] Memory stress test passes (5 min)
- [ ] Small SAM-Audio job completes without crash
- [ ] System remains responsive

**If all checked:** System is stable, proceed with full workload
**If any fail:** Check rollback plan and hardware diagnostics

---

## Next Steps After Verification

1. Update .wslconfig with new memory features
2. Test full SAM-Audio workload
3. Monitor for 24-hour stability
4. Consider gradually increasing memory speed if stable

---

**Status:** Ready for upgrade. Run PowerShell commands above.
