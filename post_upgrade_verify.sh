#!/bin/bash
# Post-WSL2 Upgrade Verification Script
# Run this immediately after upgrading WSL2 and restarting

echo "=========================================="
echo "WSL2 Upgrade Verification"
echo "=========================================="
echo ""
date
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILURES=0

# Check 1: WSL Version
echo "1. Checking WSL2 version..."
WSL_VERSION=$(wsl.exe --version 2>&1 | grep "WSL version" | head -1)
if echo "$WSL_VERSION" | grep -q "2\.[7-9]\|2\.[1-9][0-9]"; then
    echo -e "   ${GREEN}✅ SUCCESS${NC}: WSL upgraded to 2.7.0+"
    echo "   $WSL_VERSION"
else
    echo -e "   ${YELLOW}⚠️  WARNING${NC}: WSL version might not be upgraded"
    echo "   $WSL_VERSION"
    echo "   Expected: 2.7.0 or higher"
fi
echo ""

# Check 2: Config errors
echo "2. Checking for .wslconfig errors..."
# We can't directly check Windows Event Log, but we can check dmesg for startup issues
CONFIG_ERRORS=$(dmesg | grep -i "wsl.*unknown key" || echo "")
if [ -z "$CONFIG_ERRORS" ]; then
    echo -e "   ${GREEN}✅ SUCCESS${NC}: No config errors detected"
else
    echo -e "   ${RED}❌ FAILED${NC}: Config errors found:"
    echo "$CONFIG_ERRORS"
    FAILURES=$((FAILURES + 1))
fi
echo ""

# Check 3: Memory allocation
echo "3. Checking memory allocation..."
TOTAL_RAM=$(free -h | grep Mem | awk '{print $2}')
TOTAL_SWAP=$(free -h | grep Swap | awk '{print $2}')
SWAP_GB=$(free | grep Swap | awk '{print $2 / 1024 / 1024}')

echo "   Total RAM:  $TOTAL_RAM"
echo "   Total Swap: $TOTAL_SWAP"

if (( $(echo "$SWAP_GB >= 15" | bc -l) )); then
    echo -e "   ${GREEN}✅ SUCCESS${NC}: Swap configured correctly (16GB)"
else
    echo -e "   ${RED}❌ FAILED${NC}: Swap is ${TOTAL_SWAP}, expected ~16GB"
    FAILURES=$((FAILURES + 1))
fi
echo ""

# Check 4: Swap mounted
echo "4. Checking swap mount..."
SWAP_OUTPUT=$(swapon --show)
if [ -n "$SWAP_OUTPUT" ]; then
    echo -e "   ${GREEN}✅ SUCCESS${NC}: Swap is mounted"
    echo "$SWAP_OUTPUT" | sed 's/^/   /'
else
    echo -e "   ${RED}❌ FAILED${NC}: Swap not mounted"
    FAILURES=$((FAILURES + 1))
fi
echo ""

# Check 5: GPU detection
echo "5. Checking GPU detection..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1)
    if [ $? -eq 0 ]; then
        echo -e "   ${GREEN}✅ SUCCESS${NC}: GPU detected"
        echo "   $GPU_INFO"
    else
        echo -e "   ${RED}❌ FAILED${NC}: nvidia-smi error"
        echo "   $GPU_INFO"
        FAILURES=$((FAILURES + 1))
    fi
else
    echo -e "   ${YELLOW}⚠️  WARNING${NC}: nvidia-smi not found"
fi
echo ""

# Check 6: Python environment
echo "6. Checking Python environment..."
if [ -d "/home/longboardfella/sam-audio/.venv" ]; then
    echo -e "   ${GREEN}✅ SUCCESS${NC}: Virtual environment exists"
    source /home/longboardfella/sam-audio/.venv/bin/activate

    # Check Python
    PYTHON_VER=$(python --version 2>&1)
    echo "   Python: $PYTHON_VER"

    # Check PyTorch
    if python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
        TORCH_INFO=$(python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>&1)
        echo -e "   ${GREEN}✅ SUCCESS${NC}: $TORCH_INFO"
    else
        echo -e "   ${RED}❌ FAILED${NC}: PyTorch import error"
        FAILURES=$((FAILURES + 1))
    fi

    deactivate
else
    echo -e "   ${RED}❌ FAILED${NC}: Virtual environment not found"
    FAILURES=$((FAILURES + 1))
fi
echo ""

# Check 7: System errors
echo "7. Checking for recent system errors..."
ERROR_COUNT=$(dmesg | tail -100 | grep -E "error|fail|oom|crash" -i | grep -v "dxgk" | wc -l)
if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "   ${GREEN}✅ SUCCESS${NC}: No critical errors in dmesg"
else
    echo -e "   ${YELLOW}⚠️  WARNING${NC}: Found $ERROR_COUNT potential errors in dmesg"
    echo "   Recent errors:"
    dmesg | tail -100 | grep -E "error|fail|oom|crash" -i | grep -v "dxgk" | tail -5 | sed 's/^/   /'
fi
echo ""

# Check 8: Disk space
echo "8. Checking disk space..."
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 90 ]; then
    echo -e "   ${GREEN}✅ SUCCESS${NC}: Disk usage: ${DISK_USAGE}%"
else
    echo -e "   ${YELLOW}⚠️  WARNING${NC}: Disk usage high: ${DISK_USAGE}%"
fi
echo ""

# Summary
echo "=========================================="
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}STATUS: ✅ ALL CHECKS PASSED${NC}"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Review the WSL2_UPGRADE_PLAN.md"
    echo "2. Run stability tests: bash stability_test.sh"
    echo "3. Test SAM-Audio with small job"
else
    echo -e "${RED}STATUS: ❌ $FAILURES CHECK(S) FAILED${NC}"
    echo "=========================================="
    echo ""
    echo "Action required:"
    echo "1. Review failures above"
    echo "2. Check WSL2_UPGRADE_PLAN.md for troubleshooting"
    echo "3. Do NOT proceed to stability tests until fixed"
fi
echo ""

exit $FAILURES
