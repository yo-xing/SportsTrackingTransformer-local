#!/bin/bash
# Run the full pipeline for transformer model only (skips zoo model training)
# This is memory-efficient and suitable for running on full 18-week dataset

set -e  # Exit on any error

echo "========================================="
echo "Transformer-Only Training Pipeline"
echo "========================================="
echo ""

# Stage 1: Prepare extra data (with all 18 weeks)
echo "Stage 1/3: Preparing extra data (18 weeks)..."
uv run python src/prep_extra_data.py --weeks 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18
echo "✓ Data preparation complete"
echo ""

# Stage 2: Precompute datasets (both zoo and transformer needed for dependencies)
echo "Stage 2/3: Precomputing feature transforms..."
uv run dvc repro precompute_extra_datasets
echo "✓ Feature precomputation complete"
echo ""

# Stage 3: Train transformer models only
echo "Stage 3/3: Training transformer models..."
uv run dvc repro train_transformer_models
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
