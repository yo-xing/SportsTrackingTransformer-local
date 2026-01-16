#!/usr/bin/env python3
"""
Setup Inference Environment

This script checks for required files locally and copies them from Google Drive if missing.
Run this ONE TIME before running inference.

Following the logic from run_transformer_only.py to locate Google Drive caches.

Usage:
    python setup_inference.py
"""

import shutil
from pathlib import Path
import sys


def check_google_drive_mounted():
    """Check if Google Drive is mounted (Colab only)."""
    drive_path = Path("/content/drive/MyDrive")

    if not drive_path.exists():
        print("\n" + "="*60)
        print("Google Drive Not Mounted")
        print("="*60)
        print("\nThis script needs Google Drive to be mounted.")
        print("Please run the following in a Colab cell first:")
        print()
        print("    from google.colab import drive")
        print("    drive.mount('/content/drive')")
        print()
        sys.exit(1)


def setup_inference_files():
    """Check for required files and copy from Google Drive if missing."""

    print("\n" + "="*60)
    print("Inference Environment Setup")
    print("="*60 + "\n")

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    # Google Drive paths (from run_transformer_only.py)
    DRIVE_BASE = Path("/content/drive/MyDrive/SportsTrackingTransformer")
    DRIVE_CACHE_7FEAT = Path("/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_7feat_norm_football")
    MODEL_DRIVE_PATH = DRIVE_BASE / "models_norm_football/transformer/M64_L4_LR1e-04/checkpoints/epoch=41-val_loss=2.730.ckpt"

    # Local paths
    MODEL_LOCAL_PATH = script_dir / "model/epoch=41-val_loss=2.730.ckpt"
    DATASETS_LOCAL_DIR = repo_root / "data/datasets_extra_norm_football/transformer"

    all_present = True
    total_copied = 0

    # Check 1: Model checkpoint
    print("[1/2] Checking model checkpoint...")
    if MODEL_LOCAL_PATH.exists():
        size_mb = MODEL_LOCAL_PATH.stat().st_size / (1024 * 1024)
        print(f"✓ Model checkpoint exists: {MODEL_LOCAL_PATH.name} ({size_mb:.2f} MB)")
    else:
        print(f"❌ Model checkpoint not found locally")
        # Try to copy from Google Drive
        if MODEL_DRIVE_PATH.exists():
            print(f"   📁 Found in Google Drive, copying...")
            MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(MODEL_DRIVE_PATH, MODEL_LOCAL_PATH)
            size_mb = MODEL_LOCAL_PATH.stat().st_size / (1024 * 1024)
            print(f"   ✓ Copied successfully ({size_mb:.2f} MB)")
            total_copied += 1
        else:
            print(f"   ❌ Not found in Google Drive: {MODEL_DRIVE_PATH}")
            print(f"   Please download manually and place in: {MODEL_LOCAL_PATH.parent}")
            all_present = False

    print()

    # Check 2: Filtered datasets (7 features)
    print("[2/2] Checking datasets (7 features)...")
    splits = ["train", "val", "test"]
    model_type = "transformer"

    for split in splits:
        local_path = DATASETS_LOCAL_DIR / f"{split}_dataset.pkl"
        drive_path = DRIVE_CACHE_7FEAT / model_type / f"{split}_dataset.pkl"

        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"✓ {split}: {local_path.name} ({size_mb:.1f} MB)")
        elif drive_path.exists():
            size_mb = drive_path.stat().st_size / (1024 * 1024)
            print(f"📁 {split}: Found in Google Drive ({size_mb:.1f} MB)")
            print(f"   Copying to {local_path}...")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_path, local_path)
            print(f"   ✓ Copied successfully")
            total_copied += 1
        else:
            print(f"❌ {split}: Not found locally or in Drive")
            print(f"   Expected: {drive_path}")
            all_present = False

    print()

    # Summary
    print("="*60)
    if all_present:
        print("✓ All required files are present!")
        if total_copied > 0:
            print(f"✓ Copied {total_copied} file(s) from Google Drive")
        print("="*60)
        print("\nSetup complete! You can now run inference:")
        print("    cd inference")
        print("    python run_inference.py")
    else:
        print("❌ Some files are missing")
        print("="*60)
        print("\nPlease resolve the missing files before running inference.")
        print("\nExpected Google Drive locations:")
        print(f"  Model: {MODEL_DRIVE_PATH}")
        print(f"  Datasets: {DRIVE_CACHE_7FEAT}/transformer/{{train,val,test}}_dataset.pkl")
        sys.exit(1)


if __name__ == "__main__":
    # Check if on Colab with Drive mounted
    if Path("/content").exists():
        check_google_drive_mounted()

    # Check and copy files
    setup_inference_files()
