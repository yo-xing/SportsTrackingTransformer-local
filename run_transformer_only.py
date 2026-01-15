#!/usr/bin/env python3
"""
Run the full pipeline for transformer model with 7 features (distanceToGoal only).
This script uses pre-filtered datasets from the add-game-state-features branch.

The 7 features are:
- x_rel, y_rel, vx, vy, side, is_ball_carrier (6 player features)
- distanceToGoal (1 game state feature)

Usage:
    python run_transformer_only.py [--skip-filter] [--skip-training] [--skip-existing]

Options:
    --skip-filter      Skip dataset filtering (assumes datasets are already filtered)
    --skip-training    Skip training (only generate results from existing checkpoints)
    --skip-existing    Skip training models that already have checkpoints
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
    parser = ArgumentParser(description="Run transformer training pipeline with filtered features")
    # Legacy flags for consistency with other branches
    parser.add_argument("--skip-prep", action="store_true", help="Skip data preparation (not used in this branch)")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip dataset filtering step")
    parser.add_argument("--sample", type=float, default=None, help="Sample fraction of data (not used - datasets pre-filtered)")

    # Standard flags
    parser.add_argument("--skip-filter", action="store_true", help="Skip dataset filtering step (same as --skip-precompute)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (only for testing)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip training models with existing checkpoints")
    parser.add_argument("--device", type=int, default=0, help="GPU device to use (default: 0)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--use-uv", action="store_true", help="Use uv run for subprocess commands")
    args = parser.parse_args()

    skip_existing_flag = "--skip-existing" if args.skip_existing else ""
    python_cmd = "uv run python" if args.use_uv else "python"

    # --skip-precompute is an alias for --skip-filter
    skip_filter = args.skip_filter or args.skip_precompute

    print("\n" + "="*60)
    print("23-Entity Transformer Training Pipeline")
    print("Features: 7 (6 player + 1 game state)")
    print("Game State: distanceToGoal only")
    if args.sample:
        print(f"Note: --sample {args.sample} flag ignored (datasets are pre-filtered)")
    if args.skip_prep:
        print("Note: --skip-prep flag ignored (no prep step in this branch)")
    print("="*60 + "\n")

    # Step 1: Filter datasets from 11 features to 7 features
    if not skip_filter:
        run_command(
            f"{python_cmd} filter_features.py",
            "Step 1: Filtering datasets (11 features → 7 features)"
        )
    else:
        print("\n⏭️  Skipping dataset filtering (--skip-precompute)\n")

    # Step 2: Train transformer model
    if not args.skip_training:
        run_command(
            f"{python_cmd} src/train.py --model_type transformer --device {args.device} --patience {args.patience} {skip_existing_flag}",
            "Step 2: Training transformer model"
        )
    else:
        print("\n⏭️  Skipping training (--skip-training)\n")

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60 + "\n")
    print("Next steps:")
    print("  - Check tensorboard logs: tensorboard --logdir models/transformer")
    print("  - View results: models/transformer/M{dim}_L{layers}_LR{lr}/checkpoints/*.results.parquet")
    print("  - Generate summary: python src/generate_results_summary.py")
    print()


if __name__ == "__main__":
    main()
