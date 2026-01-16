#!/usr/bin/env python3
"""
Simple Inference Script for 23-Entity Football Model

This script runs the complete inference pipeline on sample data.
Based on run_transformer_only.py but simplified for inference only.

Usage:
    python run_inference.py [--model-checkpoint PATH] [--data-file PATH] [--use-uv]

The script will:
1. Check for model checkpoint in inference/model/
2. Check for sample data in inference/sample_data/Week 01/ (or custom --data-file)
3. Run data preprocessing (create_sample_data.py) if needed
4. Filter features to 7 (filter_features.py) if needed
5. Run inference (src/generate_results_summary.py)
6. Save predictions to inference/predictions/

By default, the script processes only the sample data in inference/sample_data/.
Use --data-file to specify a custom parquet file to process instead.

The preprocessing steps are run automatically if the required data files
don't exist yet. Once data is preprocessed, subsequent runs will skip
those steps.
"""

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
import shutil


# Paths
INFERENCE_DIR = Path(__file__).parent
SAMPLE_DATA_DIR = INFERENCE_DIR / "sample_data"
MODEL_DIR = INFERENCE_DIR / "model"
PREDICTIONS_DIR = INFERENCE_DIR / "predictions"


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


def find_checkpoint():
    """Find the model checkpoint in inference/model/"""
    ckpt_files = list(MODEL_DIR.glob("*.ckpt"))

    if not ckpt_files:
        print("❌ Error: No .ckpt files found in inference/model/")
        print("   Please download the model checkpoint and place it in inference/model/")
        print("   Expected file: epoch=41-val_loss=2.730.ckpt (2.49 MB)")
        print()
        print("   Download from Google Drive:")
        print("   /content/drive/MyDrive/SportsTrackingTransformer/models_norm_football/")
        print("   transformer/M64_L4_LR1e-04/checkpoints/epoch=41-val_loss=2.730.ckpt")
        sys.exit(1)

    if len(ckpt_files) > 1:
        print(f"⚠️  Found {len(ckpt_files)} checkpoint files:")
        for f in ckpt_files:
            print(f"     - {f.name}")
        print(f"   Using: {ckpt_files[0].name}")
        print()

    return ckpt_files[0]


def check_sample_data():
    """Check if sample data exists."""
    # Check for raw data
    raw_data_dir = SAMPLE_DATA_DIR / "Week 01"
    if raw_data_dir.exists():
        parquet_files = list(raw_data_dir.glob("*.parquet"))
        if parquet_files:
            print("✓ Found raw sample data:")
            for f in parquet_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    {f.name} ({size_mb:.1f} MB)")
            return True

    print("❌ Error: No sample data found in inference/sample_data/Week 01/")
    print("   Expected: .parquet files in Axially format")
    sys.exit(1)


def main():
    parser = ArgumentParser(description="Run inference on sample data")
    parser.add_argument("--model-checkpoint", type=str, default=None,
                       help="Path to model checkpoint (default: auto-detect in inference/model/)")
    parser.add_argument("--data-file", type=str, default=None,
                       help="Path to specific parquet file (default: use inference/sample_data/Week 01/)")
    parser.add_argument("--use-uv", action="store_true",
                       help="Use uv run for subprocess commands")
    args = parser.parse_args()

    python_cmd = "uv run python" if args.use_uv else "python"

    print("\n" + "="*60)
    print("Sports Tracking Transformer - Simple Inference")
    print("="*60)
    print("Model: 23-entity-football (Min+Football)")
    print("Features: 7 (6 entity + distanceToGoal)")
    print("Entities: 22 players + 1 football")
    print("="*60 + "\n")

    # Step 1: Check for model checkpoint
    print("[1/6] Checking model checkpoint...")
    if args.model_checkpoint:
        checkpoint_path = Path(args.model_checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
    else:
        checkpoint_path = find_checkpoint()

    print(f"✓ Found checkpoint: {checkpoint_path.name}")
    print()

    # Step 2: Check for sample data
    print("[2/6] Checking sample data...")
    check_sample_data()
    print()

    # Step 3: Run data preparation pipeline
    print("[3/6] Preparing data...")

    # Check if preprocessed data exists
    prepped_data_dir = Path("data/split_prepped_data_extra")
    required_files = [
        prepped_data_dir / "train_features.parquet",
        prepped_data_dir / "train_targets.parquet",
        prepped_data_dir / "val_features.parquet",
        prepped_data_dir / "val_targets.parquet",
        prepped_data_dir / "test_features.parquet",
        prepped_data_dir / "test_targets.parquet",
    ]

    missing_files = [f for f in required_files if not f.exists()]

    if missing_files:
        print("   Preprocessed data not found. Running create_sample_data.py...")
        # Run inference-specific preprocessing (processes only sample_data/)
        repo_root = INFERENCE_DIR.parent
        data_file_arg = f" --data-file {args.data_file}" if args.data_file else ""
        run_command(
            f"cd {repo_root} && {python_cmd} inference/create_sample_data.py{data_file_arg}",
            "Step 3a: Preprocessing sample data (Axially → BDB 2024 format)"
        )
    else:
        print("✓ Preprocessed data already exists")
        print()

    # Step 4: Filter features (11 features → 7 features)
    print("[4/6] Filtering features...")

    # Check if filtered datasets exist
    filtered_data_dir = Path("data/datasets_extra_norm_football/transformer")
    filtered_required = [
        filtered_data_dir / "train_dataset.pkl",
        filtered_data_dir / "val_dataset.pkl",
        filtered_data_dir / "test_dataset.pkl",
    ]

    missing_filtered = [f for f in filtered_required if not f.exists()]

    if missing_filtered:
        print("   Filtered datasets not found. Running filter_features.py...")
        # Need to run from repository root, not inference/
        repo_root = INFERENCE_DIR.parent
        run_command(
            f"cd {repo_root} && {python_cmd} filter_features.py",
            "Step 4a: Filtering datasets (11 features → 7 features)"
        )
    else:
        print("✓ Filtered datasets already exist")
        print()

    # Step 5: Prepare for inference
    print("[5/6] Setting up inference...")

    # Create predictions output directory
    PREDICTIONS_DIR.mkdir(exist_ok=True, parents=True)

    # Setup: Copy checkpoint to expected location for generate_results_summary.py
    # It expects models in models_norm_football/transformer/M{}_L{}_LR{}/checkpoints/
    temp_model_dir = Path("models_norm_football/transformer/M64_L4_LR1e-04/checkpoints")
    temp_model_dir.mkdir(parents=True, exist_ok=True)
    temp_checkpoint = temp_model_dir / checkpoint_path.name

    # Copy checkpoint if not already there
    if not temp_checkpoint.exists():
        print(f"   Copying checkpoint to temporary location...")
        shutil.copy2(checkpoint_path, temp_checkpoint)

    # Step 6: Run inference using generate_results_summary.py
    print("[6/6] Running inference...")

    # Need to run from repository root, not inference/
    repo_root = INFERENCE_DIR.parent
    run_command(
        f"cd {repo_root} && {python_cmd} src/generate_results_summary.py "
        f"--models-dir models_norm_football "
        f"--prepped-data-dir data/split_prepped_data_extra",
        "Generating results summary"
    )

    # Copy results to inference/predictions/
    results_dir = Path("models_norm_football/transformer/M64_L4_LR1e-04/checkpoints")
    result_files = list(results_dir.glob("*.results.parquet"))

    if result_files:
        print("\n✓ Copying results to inference/predictions/...")
        for result_file in result_files:
            dest = PREDICTIONS_DIR / result_file.name
            shutil.copy2(result_file, dest)
            print(f"     {result_file.name}")
        print()

    # Print summary
    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60 + "\n")
    print("Results saved to:")
    print(f"  {PREDICTIONS_DIR}/")
    print()
    print("Result files contain predictions for each split (train/val/test)")
    print()
    print("To analyze results, use the visualize.ipynb notebook:")
    print("  jupyter notebook inference/visualize.ipynb")
    print()


if __name__ == "__main__":
    main()
