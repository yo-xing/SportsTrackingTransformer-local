#!/usr/bin/env python3
"""
Run the full pipeline for transformer model only (skips zoo model training).
This is memory-efficient and suitable for running on full 18-week dataset.

Usage:
    python run_transformer_only.py [--force] [--skip-prep] [--skip-precompute]

Options:
    --force            Force recompute all stages (ignore cache)
    --skip-prep        Skip data preparation stage (use existing data)
    --skip-precompute  Skip feature precomputation stage (use existing features)
"""

import subprocess
import sys
from argparse import ArgumentParser


def run_command(cmd: str, description: str):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✓ {description} complete\n")


def main():
    parser = ArgumentParser(description="Run transformer-only training pipeline")
    parser.add_argument("--force", action="store_true", help="Force recompute all stages")
    parser.add_argument("--skip-prep", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip feature precomputation")
    args = parser.parse_args()

    force_flag = "--force" if args.force else ""

    print("\n" + "="*60)
    print("Transformer-Only Training Pipeline")
    print("="*60 + "\n")

    # Stage 1: Prepare extra data
    if not args.skip_prep:
        run_command(
            f"uv run dvc repro prep_extra_data {force_flag}",
            "Stage 1/3: Preparing extra data"
        )
    else:
        print("\n⏭️  Skipping data preparation stage\n")

    # Stage 2: Precompute datasets
    if not args.skip_precompute:
        run_command(
            f"uv run dvc repro precompute_extra_datasets {force_flag}",
            "Stage 2/3: Precomputing feature transforms"
        )
    else:
        print("\n⏭️  Skipping feature precomputation stage\n")

    # Stage 3: Train transformer models only
    run_command(
        f"uv run dvc repro train_transformer_models {force_flag}",
        "Stage 3/3: Training transformer models"
    )

    print("\n" + "="*60)
    print("Pipeline Complete! 🎉")
    print("="*60 + "\n")
    print("Trained models are in: models/transformer/\n")
    print("Next steps:")
    print("  - To pick best model: uv run dvc repro pick_best_models")
    print("  - To generate results: uv run dvc repro generate_results\n")


if __name__ == "__main__":
    main()
