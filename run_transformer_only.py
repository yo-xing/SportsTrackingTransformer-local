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
from pathlib import Path
import shutil


def check_and_sync_files():
    """
    Check for required files locally and sync from Google Drive if missing.

    This function scans for:
    1. Source datasets (11 features) - needed for filtering step
    2. Filtered datasets (7 features) - needed for training step

    Returns tuple of (source_files_exist, filtered_files_exist)
    """
    # Google Drive paths
    INPUT_DRIVE_DIR = Path("/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_gamestate_norm")
    OUTPUT_DRIVE_DIR = Path("/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_7feat_norm_football")

    # Local paths
    INPUT_DIR = Path("data/datasets_extra_gamestate_23")
    OUTPUT_DIR = Path("data/datasets_extra_norm_football")

    splits = ["train", "val", "test"]
    model_type = "transformer"

    print("\n" + "="*60)
    print("Checking for required files...")
    print("="*60 + "\n")

    # Check source datasets (11 features)
    source_files_exist = True
    source_local_count = 0
    source_drive_count = 0

    print("Source datasets (11 features):")
    for split in splits:
        local_path = INPUT_DIR / model_type / f"{split}_dataset.pkl"
        drive_path = INPUT_DRIVE_DIR / model_type / f"{split}_dataset.pkl"

        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Local: {local_path} ({size_mb:.1f} MB)")
            source_local_count += 1
        elif drive_path.exists():
            size_mb = drive_path.stat().st_size / (1024 * 1024)
            print(f"  📁 Drive: {drive_path} ({size_mb:.1f} MB)")
            print(f"     → Copying to {local_path}...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_path, local_path)
            print(f"     ✓ Copied successfully")
            source_local_count += 1
            source_drive_count += 1
        else:
            print(f"  ❌ Missing: {split}_dataset.pkl (not in local or Drive)")
            source_files_exist = False

    print()

    # Check filtered datasets (7 features)
    filtered_files_exist = True
    filtered_local_count = 0
    filtered_drive_count = 0

    print("Filtered datasets (7 features):")
    for split in splits:
        local_path = OUTPUT_DIR / model_type / f"{split}_dataset.pkl"
        drive_path = OUTPUT_DRIVE_DIR / model_type / f"{split}_dataset.pkl"

        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Local: {local_path} ({size_mb:.1f} MB)")
            filtered_local_count += 1
        elif drive_path.exists():
            size_mb = drive_path.stat().st_size / (1024 * 1024)
            print(f"  📁 Drive: {drive_path} ({size_mb:.1f} MB)")
            print(f"     → Copying to {local_path}...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_path, local_path)
            print(f"     ✓ Copied successfully")
            filtered_local_count += 1
            filtered_drive_count += 1
        else:
            print(f"  ⚠️  Missing: {split}_dataset.pkl (will be created by filtering)")
            filtered_files_exist = False

    print()

    # Summary
    if source_drive_count > 0:
        print(f"✓ Synced {source_drive_count} source dataset(s) from Drive")
    if filtered_drive_count > 0:
        print(f"✓ Synced {filtered_drive_count} filtered dataset(s) from Drive")

    if source_local_count == 3:
        print(f"✓ All source datasets available locally ({source_local_count}/3)")
    elif source_files_exist:
        print(f"⚠️  Some source datasets available locally ({source_local_count}/3)")
    else:
        print(f"❌ Source datasets missing - filtering will fail")

    if filtered_local_count == 3:
        print(f"✓ All filtered datasets available locally ({filtered_local_count}/3)")
    elif filtered_files_exist:
        print(f"⚠️  Some filtered datasets available locally ({filtered_local_count}/3)")
    else:
        print(f"⚠️  Filtered datasets missing - will be created by filtering step")

    print()

    # Check target files (needed by dataset loading)
    TARGET_DIR = Path("data/split_prepped_data_extra")
    TARGET_DRIVE_DIR = Path("/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_gamestate_norm")

    targets_exist = True
    targets_local_count = 0
    targets_drive_count = 0

    print("Target files (needed for training):")
    for split in splits:
        local_path = TARGET_DIR / f"{split}_targets.parquet"
        drive_path = TARGET_DRIVE_DIR / f"{split}_targets.parquet"

        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Local: {local_path} ({size_mb:.1f} MB)")
            targets_local_count += 1
        elif drive_path.exists():
            size_mb = drive_path.stat().st_size / (1024 * 1024)
            print(f"  📁 Drive: {drive_path} ({size_mb:.1f} MB)")
            print(f"     → Copying to {local_path}...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_path, local_path)
            print(f"     ✓ Copied successfully")
            targets_local_count += 1
            targets_drive_count += 1
        else:
            print(f"  ❌ Missing: {split}_targets.parquet (not in local or Drive)")
            targets_exist = False

    print()

    if targets_drive_count > 0:
        print(f"✓ Synced {targets_drive_count} target file(s) from Drive")

    if targets_local_count == 3:
        print(f"✓ All target files available locally ({targets_local_count}/3)")
    else:
        print(f"⚠️  Target files missing - training will fail ({targets_local_count}/3)")

    print()

    return source_files_exist, filtered_files_exist, targets_exist


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
    parser.add_argument("--skip-if-trained", action="store_true", help="Skip training models with checkpoints >= min_epochs")
    parser.add_argument("--skip-l1", action="store_true", help="Skip training all L1 (1-layer) models")
    parser.add_argument("--skip-models", type=str, default="", help="Comma-separated list of models to skip (e.g., 'M32_L2,M64_L4')")
    parser.add_argument("--min-epochs", type=int, default=40, help="Minimum epochs for --skip-if-trained (default: 40)")
    parser.add_argument("--device", type=int, default=0, help="GPU device to use (default: 0)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--use-uv", action="store_true", help="Use uv run for subprocess commands")
    args = parser.parse_args()

    skip_existing_flag = "--skip-existing" if args.skip_existing else ""
    skip_if_trained_flag = f"--skip-if-trained --min-epochs {args.min_epochs}" if args.skip_if_trained else ""
    skip_l1_flag = "--skip-l1" if args.skip_l1 else ""
    skip_models_flag = f"--skip-models {args.skip_models}" if args.skip_models else ""
    python_cmd = "uv run python" if args.use_uv else "python"

    # --skip-precompute is an alias for --skip-filter
    skip_filter = args.skip_filter or args.skip_precompute

    print("\n" + "="*60)
    print("23-Entity (with Football) Transformer Training Pipeline")
    print("Features: 7 (6 entity + 1 game state)")
    print("Entities: 22 players + 1 football")
    print("Game State: distanceToGoal only")
    if args.sample:
        print(f"Note: --sample {args.sample} flag ignored (datasets are pre-filtered)")
    if args.skip_prep:
        print("Note: --skip-prep flag ignored (no prep step in this branch)")
    print("="*60 + "\n")

    # Check and sync files from Google Drive
    source_files_exist, filtered_files_exist, targets_exist = check_and_sync_files()

    # Check for target files
    if not targets_exist:
        print("❌ Error: Target files not found")
        print("   Cannot proceed without target files")
        print("   Please ensure target files are available on Google Drive at:")
        print("   /content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_gamestate_norm/")
        sys.exit(1)

    # Determine if we can skip filtering
    if filtered_files_exist and skip_filter:
        print("✓ Filtered datasets available, skipping filtering step\n")
    elif not source_files_exist and not skip_filter:
        print("❌ Error: Source datasets (11 features) not found")
        print("   Cannot proceed with filtering step")
        print("   Please ensure datasets are available on Google Drive at:")
        print("   /content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_gamestate_norm/transformer/")
        sys.exit(1)

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
        # Verify filtered datasets exist before training
        OUTPUT_DIR_CHECK = Path("data/datasets_extra_norm_football")
        required_files = [
            OUTPUT_DIR_CHECK / "transformer" / f"{split}_dataset.pkl"
            for split in ["train", "val", "test"]
        ]
        missing_files = [f for f in required_files if not f.exists()]

        if missing_files:
            print("❌ Error: Required filtered datasets missing for training:")
            for f in missing_files:
                print(f"   - {f}")
            print("\nPlease run without --skip-filter to generate filtered datasets")
            sys.exit(1)

        run_command(
            f"{python_cmd} src/train.py --model_type transformer --device {args.device} --patience {args.patience} {skip_existing_flag} {skip_if_trained_flag} {skip_l1_flag} {skip_models_flag}",
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
