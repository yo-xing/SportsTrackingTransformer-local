#!/bin/bash
# Setup Inference Environment
# This script checks for required files and copies them from Google Drive if missing.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo "Inference Environment Setup"
echo "============================================================"
echo ""

# Google Drive paths (Colab)
DRIVE_BASE="/content/drive/MyDrive/SportsTrackingTransformer"
MODEL_DRIVE_PATH="$DRIVE_BASE/models_norm_football/transformer/M64_L4_LR1e-04/checkpoints/epoch=41-val_loss=2.730.ckpt"

# Local paths
MODEL_LOCAL_PATH="$SCRIPT_DIR/model/epoch=41-val_loss=2.730.ckpt"
SAMPLE_DATA_PATH="$SCRIPT_DIR/sample_data/Week 01/58503.parquet"

ALL_PRESENT=true

# Check 1: Model checkpoint
echo "[1/2] Checking model checkpoint..."
if [ -f "$MODEL_LOCAL_PATH" ]; then
    SIZE=$(du -h "$MODEL_LOCAL_PATH" | cut -f1)
    echo -e "${GREEN}✓ Model checkpoint exists: $(basename "$MODEL_LOCAL_PATH") ($SIZE)${NC}"
else
    echo -e "${RED}❌ Model checkpoint not found: $MODEL_LOCAL_PATH${NC}"

    # Try to copy from Google Drive
    if [ -f "$MODEL_DRIVE_PATH" ]; then
        echo "   Found in Google Drive, copying..."
        mkdir -p "$(dirname "$MODEL_LOCAL_PATH")"
        cp "$MODEL_DRIVE_PATH" "$MODEL_LOCAL_PATH"
        SIZE=$(du -h "$MODEL_LOCAL_PATH" | cut -f1)
        echo -e "   ${GREEN}✓ Copied successfully ($SIZE)${NC}"
    else
        echo -e "   ${RED}❌ Not found in Google Drive: $MODEL_DRIVE_PATH${NC}"
        echo "   Please download manually and place in: $(dirname "$MODEL_LOCAL_PATH")"
        ALL_PRESENT=false
    fi
fi

echo ""

# Check 2: Sample data
echo "[2/2] Checking sample data..."
if [ -f "$SAMPLE_DATA_PATH" ]; then
    SIZE=$(du -h "$SAMPLE_DATA_PATH" | cut -f1)
    echo -e "${GREEN}✓ Sample data exists: $(basename "$SAMPLE_DATA_PATH") ($SIZE)${NC}"
else
    echo -e "${RED}❌ Sample data not found: $SAMPLE_DATA_PATH${NC}"
    echo "   Sample data should be included in the repository"
    echo "   Expected location: $SAMPLE_DATA_PATH"
    ALL_PRESENT=false
fi

echo ""

# Summary
echo "============================================================"
if [ "$ALL_PRESENT" = true ]; then
    echo -e "${GREEN}✓ All required files are present!${NC}"
    echo "============================================================"
    echo ""
    echo "You can now run inference:"
    echo "    python run_inference.py"
else
    echo -e "${RED}❌ Some files are missing${NC}"
    echo "============================================================"
    echo ""
    echo "Please resolve the missing files before running inference."
    exit 1
fi
