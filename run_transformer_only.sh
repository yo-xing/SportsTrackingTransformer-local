#!/bin/bash
# Run the full pipeline for transformer model only (skips zoo model training)
# This is memory-efficient and suitable for running on full 18-week dataset

set -e  # Exit on any error

echo "========================================="
echo "Transformer-Only Training Pipeline"
echo "========================================="
echo ""

# Stage 1: Prepare extra data
echo "Stage 1/3: Preparing extra data..."
uv run dvc repro prep_extra_data
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
