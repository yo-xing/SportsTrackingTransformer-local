#!/usr/bin/env python3
"""
Run the full pipeline for transformer model only (skips zoo model training).
This is memory-efficient and suitable for running on full dataset.

Features: 7 (6 player + 1 game state: distanceToGoal)
Branch: 23-entity-full

Usage:
    python run_transformer_only.py [--force] [--skip-prep] [--skip-precompute] [--skip-training] [--skip-existing]

Options:
    --force            Force recompute all stages (ignore cache)
    --skip-prep        Skip data preparation stage (use existing data)
    --skip-precompute  Skip feature precomputation stage (use existing features)
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
    parser = ArgumentParser(description="Run transformer-only training pipeline")
    parser.add_argument("--force", action="store_true", help="Force recompute all stages (deletes caches)")
    parser.add_argument("--skip-prep", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip feature precomputation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (only generate results from existing checkpoints)")
    parser.add_argument("--sample", type=float, default=None, help="Sample fraction of data (e.g., 0.1 for 10%%)")
    parser.add_argument("--retrain", action="store_true", help="Delete existing model checkpoints before training")
    parser.add_argument("--skip-existing", action="store_true", help="Skip training models that already have checkpoints")
    parser.add_argument("--skip-if-trained", action="store_true", help="Skip training models with checkpoints >= min_epochs")
    parser.add_argument("--min-epochs", type=int, default=40, help="Minimum epochs for --skip-if-trained (default: 40)")
    parser.add_argument("--device", type=int, default=0, help="GPU device to use (default: 0)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training (default: 256)")
    args = parser.parse_args()

    force_flag = "--force" if args.force else ""
    sample_flag = f"--sample {args.sample}" if args.sample else ""
    skip_existing_flag = "--skip-existing" if args.skip_existing else ""
    skip_if_trained_flag = f"--skip-if-trained --min-epochs {args.min_epochs}" if args.skip_if_trained else ""

    print("\n" + "="*60)
    print("Transformer-Only Training Pipeline")
    print("Branch: 23-entity-full")
    print("Features: 7 (6 player + 1 game state: distanceToGoal)")
    print("="*60 + "\n")

    # Use separate cache directories for 23-entity-full branch
    # This prevents overwriting existing caches from other branches
    drive_cache_prep = "/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_23entity_full"
    drive_cache_datasets = "/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_7feat_23entity_full"
    local_output_prep = "data/split_prepped_data_extra"
    local_output_datasets = "data/datasets_extra"
    local_output_models = "models_23entity_full"

    # If force flag is set, delete cache directories to force regeneration
    if args.force:
        print("\n🔄 Force flag set - deleting cache directories...\n")
        import shutil
        from pathlib import Path

        # Delete local caches
        for cache_dir in [local_output_prep, local_output_datasets]:
            if Path(cache_dir).exists():
                print(f"  Deleting {cache_dir}")
                shutil.rmtree(cache_dir)

        # Delete Google Drive caches if mounted
        for cache_dir in [drive_cache_prep, drive_cache_datasets]:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                print(f"  Deleting {cache_dir}")
                shutil.rmtree(cache_dir)

        print("\n✓ Cache directories deleted\n")

    # If retrain flag is set, delete existing model checkpoints
    if args.retrain:
        print("\n🔄 Retrain flag set - deleting existing model checkpoints...\n")
        import shutil
        from pathlib import Path

        # Delete local models
        if Path(local_output_models).exists():
            print(f"  Deleting {local_output_models}")
            shutil.rmtree(local_output_models)

        # Delete Google Drive models if mounted
        drive_models_dir = "/content/drive/MyDrive/SportsTrackingTransformer/models_23entity_full"
        drive_models_path = Path(drive_models_dir)
        if drive_models_path.exists():
            print(f"  Deleting {drive_models_dir}")
            shutil.rmtree(drive_models_dir)

        print("\n✓ Model checkpoints deleted\n")

    # Stage 1: Prepare extra data (with all available weeks)
    if not args.skip_prep:
        sample_desc = f" ({args.sample*100:.0f}% sample)" if args.sample else ""
        run_command(
            f"uv run python src/prep_extra_data.py "
            f"--output-dir {local_output_prep} --drive-dir {drive_cache_prep} {sample_flag}",
            f"Stage 1/3: Preparing extra data{sample_desc}"
        )
    else:
        print("\n⏭️  Skipping data preparation stage\n")

    # Stage 2: Precompute datasets (transformer only, skip zoo)
    if not args.skip_precompute:
        run_command(
            f"uv run python src/datasets.py "
            f"--prepped-data-dir {local_output_prep} "
            f"--dataset-dir {local_output_datasets} "
            f"--drive-dir {drive_cache_datasets}",
            "Stage 2/3: Precomputing feature transforms (transformer only)"
        )
    else:
        print("\n⏭️  Skipping feature precomputation stage\n")

    # Stage 3: Train transformer models only
    if not args.skip_training:
        # Dynamically set num_workers based on CPU count (leave some headroom)
        import os
        cpu_count = os.cpu_count() or 8
        num_workers = max(4, min(cpu_count - 4, 16))  # Use CPU count - 4, max 16, min 4

        run_command(
            f"uv run python src/train.py --model_type transformer --device {args.device} "
            f"--patience {args.patience} "
            f"--batch-size {args.batch_size} --num-workers {num_workers} "
            f"{skip_existing_flag} {skip_if_trained_flag}",
            "Stage 3/3: Training transformer models"
        )
    else:
        print("\n⏭️  Skipping training stage (using existing checkpoints)\n")

    # Models are already in Google Drive (no backup needed)
    drive_models_dir = "/content/drive/MyDrive/SportsTrackingTransformer/models_23entity_full"
    from pathlib import Path
    if Path("/content/drive/MyDrive").exists():
        print("\n" + "="*60)
        print("Models saved to Google Drive")
        print("="*60 + "\n")
        print(f"✓ Models are already in Google Drive: {drive_models_dir}/transformer/\n")
    else:
        print("\nGoogle Drive not mounted\n")

    print("\n" + "="*60)
    print("Pipeline Complete! 🎉")
    print("="*60 + "\n")
    print("Trained models are in:")
    print(f"  - Local: {local_output_models}/transformer/")
    if Path("/content/drive/MyDrive").exists():
        print(f"  - Google Drive: {drive_models_dir}/transformer/")
    print("\nCache locations:")
    print(f"  - Prep data cache: {drive_cache_prep}")
    print(f"  - Dataset cache: {drive_cache_datasets}")
    print(f"  - Local prep data: {local_output_prep}")
    print(f"  - Local datasets: {local_output_datasets}\n")
    print("Note: Feature selection (7 features) happens during dataset creation.")
    print("The transformer uses only: x_rel, y_rel, vx, vy, side, is_ball_carrier, distanceToGoal\n")
    print("Next steps:")
    print(f"  - View logs: tensorboard --logdir {local_output_models}/transformer")
    print(f"  - Check results: {local_output_models}/transformer/M{{dim}}_L{{layers}}_LR{{lr}}/checkpoints/*.results.parquet")
    print("  - Select best model: python src/pick_best_models.py")
    print("  - Generate summary: python src/generate_results_summary.py\n")


if __name__ == "__main__":
    main()
