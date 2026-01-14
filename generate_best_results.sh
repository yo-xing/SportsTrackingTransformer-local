#!/bin/bash
# Generate results for the best model checkpoint only

set -e

MODELS_DIR="${1:-/content/drive/MyDrive/SportsTrackingTransformer/models_gamestate}"
PREPPED_DATA_DIR="${2:-data/split_prepped_data_extra_gamestate}"
NUM_FEATURES="${3:-11}"

echo "Finding best checkpoint in $MODELS_DIR..."

# Find best checkpoint (lowest val_loss)
BEST_CKPT=$(find "$MODELS_DIR/transformer" -name "*.ckpt" -type f | \
    while read f; do
        valloss=$(echo "$f" | grep -o "val_loss=[0-9.]*" | cut -d= -f2)
        if [ ! -z "$valloss" ]; then
            echo "$valloss $f"
        fi
    done | sort -n | head -1 | cut -d' ' -f2-)

if [ -z "$BEST_CKPT" ]; then
    echo "Error: No checkpoints found"
    exit 1
fi

# Extract model info
MODEL_NAME=$(echo "$BEST_CKPT" | grep -o "M[0-9]*_L[0-9]*_LR[^/]*")
VAL_LOSS=$(echo "$BEST_CKPT" | grep -o "val_loss=[0-9.]*")

echo ""
echo "Best model: $MODEL_NAME"
echo "Validation loss: $VAL_LOSS"
echo "Checkpoint: $BEST_CKPT"
echo ""

# Check if results already exist
RESULTS_FILE="${BEST_CKPT%.ckpt}.results.parquet"
if [ -f "$RESULTS_FILE" ]; then
    echo "✓ Results already exist: $RESULTS_FILE"
    exit 0
fi

echo "Generating results..."
uv run python src/generate_results_summary.py \
    --models-dir "$MODELS_DIR" \
    --prepped-data-dir "$PREPPED_DATA_DIR" \
    --num-features "$NUM_FEATURES" \
    --best-only

echo ""
echo "✓ Done! Results saved to: $RESULTS_FILE"
