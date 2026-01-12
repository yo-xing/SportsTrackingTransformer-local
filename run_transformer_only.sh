#!/bin/bash
# Run the full pipeline for transformer model only (skips zoo model training)
# This is memory-efficient and suitable for running on full 18-week dataset
# Uses separate cache directories to avoid overwriting 2-week DVC pipeline caches

set -e  # Exit on any error

# Separate cache directories for full 18-week data
DRIVE_CACHE_PREP="/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_full18weeks"
DRIVE_CACHE_DATASETS="/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_full18weeks"
LOCAL_OUTPUT_PREP="data/split_prepped_data_extra_full18weeks"
LOCAL_OUTPUT_DATASETS="data/datasets_extra_full18weeks"
LOCAL_OUTPUT_MODELS="models_full18weeks"

echo "========================================="
echo "Transformer-Only Training Pipeline"
echo "========================================="
echo ""

# Stage 1: Prepare extra data (with all 18 weeks)
echo "Stage 1/3: Preparing extra data (18 weeks)..."
uv run python src/prep_extra_data.py \
  --weeks 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 \
  --output-dir "$LOCAL_OUTPUT_PREP" \
  --drive-dir "$DRIVE_CACHE_PREP"
echo "✓ Data preparation complete"
echo ""

# Stage 2: Precompute datasets
echo "Stage 2/3: Precomputing feature transforms..."
uv run python src/datasets.py \
  --prepped-data-dir "$LOCAL_OUTPUT_PREP" \
  --dataset-dir "$LOCAL_OUTPUT_DATASETS" \
  --drive-dir "$DRIVE_CACHE_DATASETS"
echo "✓ Feature precomputation complete"
echo ""

# Stage 3: Train transformer models only
echo "Stage 3/3: Training transformer models..."
uv run python src/train.py --model_type transformer --device 0 \
  --dataset-dir "$LOCAL_OUTPUT_DATASETS" \
  --models-dir "$LOCAL_OUTPUT_MODELS" \
  --skip-existing
echo "✓ Transformer training complete"
echo ""

echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Trained models are in: models/transformer/"
echo ""
echo "Next steps:"
echo "  - To generate results: uv run dvc repro generate_results"
echo "  - To pick best model: uv run dvc repro pick_best_models"
