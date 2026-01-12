# WSL2 Upgrade & Stability Test - Quick Start Guide

**Date:** 2026-01-11
**Status:** Ready to upgrade
**Current WSL Version:** 2.5.7.0
**Target Version:** 2.7.0+

---

## What We Fixed

1. ✅ Removed invalid config options (autoMemoryReclaim, sparseVhd) from current .wslconfig
2. ✅ Hardware reseated (RAM + NVIDIA GPU)
3. ✅ Memory speed reduced to 4600MHz
4. ✅ XMP disabled in BIOS
5. ✅ Swap working correctly (16GB)

---

## Upgrade Commands (Copy & Paste)

### Step 1: Upgrade WSL2 (Windows PowerShell as Administrator)

```powershell
# Check current version
wsl --version

# Upgrade to latest
wsl --update

# Verify upgrade
wsl --version

# Shutdown WSL2
wsl --shutdown
```

**Expected output:** WSL version: 2.7.0 or higher

---

### Step 2: Update .wslconfig (Windows PowerShell)

```powershell
# View the new config first
notepad C:\Users\paul\.wslconfig

# Or use this command to copy from WSL (after WSL restarts):
# From WSL terminal: cat /home/longboardfella/sam-audio/wslconfig_upgraded.txt
# Then manually copy to C:\Users\paul\.wslconfig in Windows

# After updating, shutdown WSL again
wsl --shutdown
```

---

### Step 3: Verify Installation (WSL Terminal)

```bash
cd /home/longboardfella/sam-audio

# Run verification script
bash post_upgrade_verify.sh
```

**Expected:** All checks should pass ✅

---

### Step 4: Run Stability Tests (WSL Terminal)

```bash
# Run comprehensive stability tests (~10-15 min)
bash stability_test.sh
```

**Expected:** All 4 tests should pass ✅

---

### Step 5: Test SAM-Audio (WSL Terminal)

```bash
# Activate environment
source .venv/bin/activate

# Run a small test job
python run_sam_audio_ultrarobust.py \
  --file mnt/f/hf-home/audio_input/Campbell_Audio.wav \
  --desc "test run after upgrade" \
  --verbose
```

**Monitor in separate terminal:**
```bash
watch -n 1 'free -h && echo --- && nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader'
```

---

## Files Created

| File | Purpose |
|------|---------|
| `WSL2_UPGRADE_PLAN.md` | Detailed upgrade plan and rollback procedures |
| `post_upgrade_verify.sh` | Verification script (run after upgrade) |
| `stability_test.sh` | Comprehensive stability tests |
| `wslconfig_upgraded.txt` | New .wslconfig with WSL 2.7+ features |
| `UPGRADE_QUICK_START.md` | This file (quick reference) |

---

## What Gets Tested

### Verification Script (`post_upgrade_verify.sh`)
- ✅ WSL2 version upgraded
- ✅ No config errors
- ✅ Memory and swap correctly allocated
- ✅ GPU detected
- ✅ Python environment intact
- ✅ PyTorch + CUDA working

### Stability Tests (`stability_test.sh`)
- **Test 1:** RAM stress (8GB for 3 min)
- **Test 2:** GPU memory allocation (10GB)
- **Test 3:** Combined CPU+RAM stress (2 min)
- **Test 4:** Memory leak detection

---

## New WSL 2.7+ Features

After upgrade, your .wslconfig will include:

```ini
autoMemoryReclaim=gradual  # Reclaim unused memory back to Windows
sparseVhd=true             # Virtual disk can shrink
```

**Benefits:**
- Better memory management
- Windows gets RAM back when WSL2 idle
- Virtual disk doesn't grow indefinitely
- More stable under heavy loads

---

## Success Criteria

Before running full workloads, verify:

- [ ] WSL 2.7.0+ installed
- [ ] No config errors on startup
- [ ] `post_upgrade_verify.sh` all checks passed
- [ ] `stability_test.sh` all tests passed
- [ ] Small SAM-Audio job completes without crash
- [ ] System remains responsive

---

## If Something Goes Wrong

### WSL2 Won't Start
```powershell
# Check Event Viewer
Get-WinEvent -LogName System -MaxEvents 50 | Where-Object {$_.LevelDisplayName -eq "Error"}

# Reset WSL2
wsl --shutdown
wsl --unregister Ubuntu
wsl --install Ubuntu
```

### Config Errors Persist
```bash
# Revert to basic config
cat > /mnt/c/Users/paul/.wslconfig << 'EOF'
[wsl2]
memory=32GB
swap=16GB
swapFile=C:\\Users\\paul\\AppData\\Local\\Temp\\wsl-swap.vhdx
processors=16
EOF
```

### Stability Tests Fail
1. Check hardware: RAM reseated properly?
2. Check BIOS: Memory speed 4600MHz? XMP disabled?
3. Run memtest86+ from BIOS (4+ hours)
4. Check Windows Event Viewer for hardware errors

---

## After Successful Upgrade

1. **First run:** Monitor with `watch -n 1 'free -h'`
2. **24-hour test:** Run typical workloads, watch for crashes
3. **If stable for 24h:** Consider gradually increasing memory speed
4. **Document:** Note any changes in performance or stability

---

## Current Hardware Configuration

- **RAM:** 64GB @ 4600MHz (XMP disabled)
- **GPU:** Quadro RTX 8000 (48GB VRAM)
- **CPU:** 24 cores total (16 to WSL, 8 to Windows)
- **WSL Memory:** 32GB allocated
- **WSL Swap:** 16GB configured

---

## Quick Status Check Anytime

```bash
# Memory
free -h

# Swap
swapon --show

# GPU
nvidia-smi

# System errors
dmesg | tail -50 | grep -E "error|oom|crash" -i
```

---

## Questions?

Refer to detailed documentation:
- `WSL2_UPGRADE_PLAN.md` - Full upgrade plan
- `WSL2_CRASH_DIAGNOSIS.md` - Previous issue analysis
- `README_WSL_CRASH.txt` - Quick crash fix reference

---

**Ready to proceed?** Start with Step 1 above (Windows PowerShell).
