#!/bin/bash

##############################################################################
# Parallel Incremental Experiments for ExiGCN
# Runs multiple datasets simultaneously on different GPUs
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "ExiGCN Parallel Incremental Experiments"
echo "======================================================================"

# Check conda environment
if [[ "$CONDA_DEFAULT_ENV" != "exigcn" ]]; then
    echo -e "${RED}ERROR: Not in exigcn conda environment!${NC}"
    echo "Please run: conda activate exigcn"
    exit 1
fi

# Create log directory
mkdir -p results/logs

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "\n${GREEN}Starting parallel experiments at $TIMESTAMP${NC}\n"

# GPU Configuration
# GPU 0: Cora-Full
# GPU 1: Reddit (largest dataset)
# GPU 2: Amazon-Computer  
# GPU 3: OGBN-arXiv

##############################################################################
# Launch experiments in background
##############################################################################

echo -e "${YELLOW}[GPU 0]${NC} Launching Cora-Full incremental experiment..."
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_incremental \
    --config configs/cora_full.yaml \
    2>&1 | tee results/logs/cora_full_inc_${TIMESTAMP}.log &
PID_CORA=$!
echo "  PID: $PID_CORA"

sleep 2

echo -e "${YELLOW}[GPU 1]${NC} Launching Reddit incremental experiment..."
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_incremental \
    --config configs/reddit.yaml \
    2>&1 | tee results/logs/reddit_inc_${TIMESTAMP}.log &
PID_REDDIT=$!
echo "  PID: $PID_REDDIT"

sleep 2

echo -e "${YELLOW}[GPU 2]${NC} Launching Amazon-Computer incremental experiment..."
CUDA_VISIBLE_DEVICES=2 python -m experiments.run_incremental \
    --config configs/amazon_computer.yaml \
    2>&1 | tee results/logs/amazon_computer_inc_${TIMESTAMP}.log &
PID_AMAZON=$!
echo "  PID: $PID_AMAZON"

sleep 2

echo -e "${YELLOW}[GPU 3]${NC} Launching OGBN-arXiv incremental experiment..."
CUDA_VISIBLE_DEVICES=3 python -m experiments.run_incremental \
    --config configs/ogbn_arxiv.yaml \
    2>&1 | tee results/logs/ogbn_arxiv_inc_${TIMESTAMP}.log &
PID_ARXIV=$!
echo "  PID: $PID_ARXIV"

echo ""
echo "======================================================================"
echo "All experiments launched!"
echo "======================================================================"
echo "Cora-Full (GPU 0):       PID $PID_CORA"
echo "Reddit (GPU 1):          PID $PID_REDDIT"
echo "Amazon-Computer (GPU 2): PID $PID_AMAZON"
echo "OGBN-arXiv (GPU 3):      PID $PID_ARXIV"
echo "======================================================================"

##############################################################################
# Monitoring
##############################################################################

echo -e "\n${GREEN}Monitoring experiments...${NC}"
echo "Press Ctrl+C to stop monitoring (experiments will continue)"
echo ""

# Function to check if process is running
is_running() {
    kill -0 $1 2>/dev/null
}

# Monitor loop
while true; do
    sleep 10
    
    # Check status
    CORA_STATUS="✓ Done"
    REDDIT_STATUS="✓ Done"
    AMAZON_STATUS="✓ Done"
    ARXIV_STATUS="✓ Done"
    
    if is_running $PID_CORA; then
        CORA_STATUS="⏳ Running"
    fi
    
    if is_running $PID_REDDIT; then
        REDDIT_STATUS="⏳ Running"
    fi
    
    if is_running $PID_AMAZON; then
        AMAZON_STATUS="⏳ Running"
    fi
    
    if is_running $PID_ARXIV; then
        ARXIV_STATUS="⏳ Running"
    fi
    
    # Clear screen and show status
    clear
    echo "======================================================================"
    echo "ExiGCN Parallel Experiments - Status"
    echo "======================================================================"
    echo "Started: $TIMESTAMP"
    echo "Current: $(date +%Y%m%d_%H%M%S)"
    echo ""
    echo "Cora-Full (GPU 0):       $CORA_STATUS"
    echo "Reddit (GPU 1):          $REDDIT_STATUS"
    echo "Amazon-Computer (GPU 2): $AMAZON_STATUS"
    echo "OGBN-arXiv (GPU 3):      $ARXIV_STATUS"
    echo ""
    echo "======================================================================"
    echo "GPU Status:"
    echo "======================================================================"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "GPU %s: %s | Util: %3s%% | Mem: %5s/%5s MB\n", $1, $2, $3, $4, $5}'
    echo "======================================================================"
    echo ""
    echo "Log files:"
    echo "  Cora-Full:       results/logs/cora_full_inc_${TIMESTAMP}.log"
    echo "  Reddit:          results/logs/reddit_inc_${TIMESTAMP}.log"
    echo "  Amazon-Computer: results/logs/amazon_computer_inc_${TIMESTAMP}.log"
    echo "  OGBN-arXiv:      results/logs/ogbn_arxiv_inc_${TIMESTAMP}.log"
    echo ""
    echo "Press Ctrl+C to stop monitoring (experiments will continue)"
    
    # Check if all done
    if ! is_running $PID_CORA && ! is_running $PID_REDDIT && ! is_running $PID_AMAZON && ! is_running $PID_ARXIV; then
        echo ""
        echo -e "${GREEN}======================================================================"
        echo "All experiments completed!"
        echo "======================================================================${NC}"
        break
    fi
    
    sleep 50  # Update every 60 seconds total
done

##############################################################################
# Summary
##############################################################################

echo ""
echo "======================================================================"
echo "Experiment Summary"
echo "======================================================================"
echo ""

# Wait for all processes to finish
wait $PID_CORA 2>/dev/null
CORA_EXIT=$?

wait $PID_REDDIT 2>/dev/null
REDDIT_EXIT=$?

wait $PID_AMAZON 2>/dev/null
AMAZON_EXIT=$?

wait $PID_ARXIV 2>/dev/null
ARXIV_EXIT=$?

# Check exit codes
if [ $CORA_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Cora-Full:       Success"
else
    echo -e "${RED}✗${NC} Cora-Full:       Failed (exit code: $CORA_EXIT)"
fi

if [ $REDDIT_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Reddit:          Success"
else
    echo -e "${RED}✗${NC} Reddit:          Failed (exit code: $REDDIT_EXIT)"
fi

if [ $AMAZON_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Amazon-Computer: Success"
else
    echo -e "${RED}✗${NC} Amazon-Computer: Failed (exit code: $AMAZON_EXIT)"
fi

if [ $ARXIV_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} OGBN-arXiv:      Success"
else
    echo -e "${RED}✗${NC} OGBN-arXiv:      Failed (exit code: $ARXIV_EXIT)"
fi

echo ""
echo "======================================================================"
echo "Results saved in:"
echo "  - results/tables/"
echo "  - results/summaries/"
echo "======================================================================"

##############################################################################
# Generate Visualizations
##############################################################################

echo ""
echo "======================================================================"
echo "Generating Visualizations..."
echo "======================================================================"

python scripts/visualize_results.py

echo ""
echo "======================================================================"
echo "Figures saved in:"
echo "  - results/figures/"
echo "======================================================================"
echo ""
echo "Finished at: $(date)"
echo "======================================================================"