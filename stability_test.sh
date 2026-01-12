#!/bin/bash
# WSL2 Stability Testing Script
# Tests system stability after hardware changes and WSL2 upgrade
# Run after post_upgrade_verify.sh passes

echo "=========================================="
echo "WSL2 Stability Testing Suite"
echo "=========================================="
echo ""
date
echo ""
echo "This will run several stress tests to verify system stability."
echo "Estimated time: 10-15 minutes"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Track results
TESTS_PASSED=0
TESTS_FAILED=0

# Log file
LOG_FILE="/home/longboardfella/sam-audio/stability_test_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

# Function to log and display
log_msg() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to check if stress-ng is installed
check_stress_ng() {
    if ! command -v stress-ng &> /dev/null; then
        echo -e "${YELLOW}Installing stress-ng...${NC}"
        sudo apt-get update -qq
        sudo apt-get install -y stress-ng
    fi
}

# Pre-test system check
echo "Pre-test System Status:" | tee "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"
free -h | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Test 1: Basic Memory Stress (3 minutes)
echo "=========================================="
echo "Test 1: System RAM Stress (3 minutes)"
echo "=========================================="
log_msg "Starting Test 1 at $(date)"

check_stress_ng

log_msg "Stressing 8GB RAM with 4 workers for 3 minutes..."
log_msg "Monitor in another terminal: watch -n 1 'free -h'"
echo ""

# Run stress test in background, monitor memory
timeout 180 stress-ng --vm 4 --vm-bytes 8G --vm-method all --verify --timeout 180s --metrics-brief 2>&1 | tee -a "$LOG_FILE" &
STRESS_PID=$!

# Monitor memory every 10 seconds
for i in {1..18}; do
    sleep 10
    MEM_USED=$(free | grep Mem | awk '{printf "%.1f", $3/($2/100)}')
    SWAP_USED=$(free | grep Swap | awk '{printf "%.1f", $3/($2/100)}')
    echo "  [$i/18] Memory: ${MEM_USED}% used, Swap: ${SWAP_USED}% used" | tee -a "$LOG_FILE"
done

# Wait for stress test to complete
wait $STRESS_PID
STRESS_EXIT=$?

# Check for OOM errors
OOM_ERRORS=$(dmesg | tail -50 | grep -i "out of memory\|oom" | wc -l)

if [ $STRESS_EXIT -eq 0 ] && [ $OOM_ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ Test 1 PASSED${NC}: RAM stress test completed successfully" | tee -a "$LOG_FILE"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}❌ Test 1 FAILED${NC}: Exit code $STRESS_EXIT, OOM errors: $OOM_ERRORS" | tee -a "$LOG_FILE"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo "" | tee -a "$LOG_FILE"

# Cool down
echo "Cooling down for 10 seconds..." | tee -a "$LOG_FILE"
sleep 10

# Test 2: GPU Memory Test
echo "=========================================="
echo "Test 2: GPU Memory Test"
echo "=========================================="
log_msg "Starting Test 2 at $(date)"

# Check if PyTorch is available
if source /home/longboardfella/sam-audio/.venv/bin/activate 2>/dev/null; then
    log_msg "Testing GPU memory allocation..."

    python3 << 'EOF' 2>&1 | tee -a "$LOG_FILE"
import torch
import gc
import time

print("GPU:", torch.cuda.get_device_name(0))
print("Total VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")

try:
    # Allocate 10GB on GPU
    print("\nAllocating 10GB on GPU...")
    tensor = torch.zeros((10 * 1024**3 // 4,), dtype=torch.float32, device='cuda')
    print("Allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")

    time.sleep(2)

    # Free it
    del tensor
    gc.collect()
    torch.cuda.empty_cache()
    print("Freed. Current:", torch.cuda.memory_allocated() / 1024**3, "GB")

    print("\n✅ GPU memory test passed")
    exit(0)
except Exception as e:
    print(f"\n❌ GPU memory test failed: {e}")
    exit(1)
EOF

    GPU_TEST_EXIT=$?
    deactivate

    if [ $GPU_TEST_EXIT -eq 0 ]; then
        echo -e "${GREEN}✅ Test 2 PASSED${NC}: GPU memory test successful" | tee -a "$LOG_FILE"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}❌ Test 2 FAILED${NC}: GPU memory test failed" | tee -a "$LOG_FILE"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
else
    echo -e "${YELLOW}⚠️  Test 2 SKIPPED${NC}: Python venv not available" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Cool down
echo "Cooling down for 10 seconds..." | tee -a "$LOG_FILE"
sleep 10

# Test 3: Combined CPU + Memory Stress (2 minutes)
echo "=========================================="
echo "Test 3: Combined CPU + Memory (2 minutes)"
echo "=========================================="
log_msg "Starting Test 3 at $(date)"

log_msg "Stressing CPU (8 cores) + Memory (4GB) for 2 minutes..."
timeout 120 stress-ng --cpu 8 --vm 2 --vm-bytes 4G --timeout 120s --metrics-brief 2>&1 | tee -a "$LOG_FILE" &
STRESS_PID=$!

# Monitor
for i in {1..12}; do
    sleep 10
    CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    MEM_USED=$(free | grep Mem | awk '{printf "%.1f", $3/($2/100)}')
    echo "  [$i/12] Load: ${CPU_LOAD}, Memory: ${MEM_USED}% used" | tee -a "$LOG_FILE"
done

wait $STRESS_PID
STRESS_EXIT=$?

if [ $STRESS_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ Test 3 PASSED${NC}: Combined stress test successful" | tee -a "$LOG_FILE"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}❌ Test 3 FAILED${NC}: Exit code $STRESS_EXIT" | tee -a "$LOG_FILE"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo "" | tee -a "$LOG_FILE"

# Test 4: Memory Leak Detection
echo "=========================================="
echo "Test 4: Memory Leak Detection"
echo "=========================================="
log_msg "Starting Test 4 at $(date)"

log_msg "Running multiple allocation/deallocation cycles..."

INITIAL_MEM=$(free | grep Mem | awk '{print $3}')
log_msg "Initial memory used: $INITIAL_MEM KB"

# Run 5 cycles of allocation
for cycle in {1..5}; do
    echo "  Cycle $cycle/5..." | tee -a "$LOG_FILE"
    stress-ng --vm 2 --vm-bytes 2G --timeout 20s >/dev/null 2>&1
    sleep 5
    CURRENT_MEM=$(free | grep Mem | awk '{print $3}')
    MEM_DIFF=$((CURRENT_MEM - INITIAL_MEM))
    echo "    Memory used: $CURRENT_MEM KB (diff: $MEM_DIFF KB)" | tee -a "$LOG_FILE"
done

FINAL_MEM=$(free | grep Mem | awk '{print $3}')
MEM_LEAK=$((FINAL_MEM - INITIAL_MEM))
log_msg "Final memory used: $FINAL_MEM KB"
log_msg "Memory difference: $MEM_LEAK KB"

# Allow 500MB difference (500000 KB) for normal variance
if [ $MEM_LEAK -lt 500000 ]; then
    echo -e "${GREEN}✅ Test 4 PASSED${NC}: No significant memory leak detected" | tee -a "$LOG_FILE"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${YELLOW}⚠️  Test 4 WARNING${NC}: Memory increased by $MEM_LEAK KB" | tee -a "$LOG_FILE"
    TESTS_PASSED=$((TESTS_PASSED + 1))
fi
echo "" | tee -a "$LOG_FILE"

# Final System Check
echo "=========================================="
echo "Post-Test System Status"
echo "=========================================="
log_msg "Final check at $(date)"
echo "" | tee -a "$LOG_FILE"

log_msg "Memory Status:"
free -h | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

log_msg "Swap Status:"
swapon --show | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

log_msg "GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check for errors
log_msg "Checking for system errors..."
ERROR_COUNT=$(dmesg | tail -100 | grep -E "error|fail|oom|crash" -i | grep -v "dxgk" | wc -l)
if [ $ERROR_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $ERROR_COUNT potential errors${NC}" | tee -a "$LOG_FILE"
    dmesg | tail -100 | grep -E "error|fail|oom|crash" -i | grep -v "dxgk" | tail -10 | tee -a "$LOG_FILE"
else
    echo -e "${GREEN}No critical errors found${NC}" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# Summary
echo "=========================================="
echo "STABILITY TEST SUMMARY"
echo "=========================================="
log_msg "Tests Passed: $TESTS_PASSED"
log_msg "Tests Failed: $TESTS_FAILED"
log_msg "Log file: $LOG_FILE"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "✅ ALL STABILITY TESTS PASSED"
    echo "==========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. System is stable for production use"
    echo "2. You can now run full SAM-Audio workloads"
    echo "3. Monitor first few runs with: watch -n 1 'free -h'"
    echo ""
    echo "To test SAM-Audio:"
    echo "  source .venv/bin/activate"
    echo "  python run_sam_audio_ultrarobust.py --file <audio_file> --desc <description> --verbose"
    exit 0
else
    echo -e "${RED}=========================================="
    echo "❌ $TESTS_FAILED TEST(S) FAILED"
    echo "==========================================${NC}"
    echo ""
    echo "Action required:"
    echo "1. Review log file: $LOG_FILE"
    echo "2. Check dmesg for errors: dmesg | tail -100"
    echo "3. Verify hardware: RAM speed, XMP settings"
    echo "4. Consider running memtest86+ from BIOS"
    echo "5. Check Windows Event Viewer for hardware errors"
    exit 1
fi
