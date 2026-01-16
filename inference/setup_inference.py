#!/usr/bin/env python3
"""
Setup Inference Environment

This script checks for required files and copies them from Google Drive if missing.
Run this script on Google Colab before running inference.

Usage:
    python setup_inference.py
"""

import shutil
from pathlib import Path
import sys


def check_and_copy_files():
    """Check for required files and copy from Google Drive if missing."""

    print("\n" + "="*60)
    print("Inference Environment Setup")
    print("="*60 + "\n")

    # Paths
    script_dir = Path(__file__).parent

    # Google Drive paths (Colab)
    DRIVE_BASE = Path("/content/drive/MyDrive/SportsTrackingTransformer")
    MODEL_DRIVE_PATH = DRIVE_BASE / "models_norm_football/transformer/M64_L4_LR1e-04/checkpoints/epoch=41-val_loss=2.730.ckpt"

    # Local paths
    MODEL_LOCAL_PATH = script_dir / "model/epoch=41-val_loss=2.730.ckpt"
    SAMPLE_DATA_PATH = script_dir / "sample_data/Week 01/58503.parquet"

    all_present = True

    # Check 1: Model checkpoint
    print("[1/2] Checking model checkpoint...")
    if MODEL_LOCAL_PATH.exists():
        size_mb = MODEL_LOCAL_PATH.stat().st_size / (1024 * 1024)
        print(f"✓ Model checkpoint exists: {MODEL_LOCAL_PATH.name} ({size_mb:.2f} MB)")
    else:
        print(f"❌ Model checkpoint not found: {MODEL_LOCAL_PATH}")

        # Try to copy from Google Drive
        if MODEL_DRIVE_PATH.exists():
            print(f"   Found in Google Drive, copying...")
            MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(MODEL_DRIVE_PATH, MODEL_LOCAL_PATH)
            size_mb = MODEL_LOCAL_PATH.stat().st_size / (1024 * 1024)
            print(f"   ✓ Copied successfully ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ Not found in Google Drive: {MODEL_DRIVE_PATH}")
            print(f"   Please download manually and place in: {MODEL_LOCAL_PATH.parent}")
            all_present = False

    print()

    # Check 2: Sample data
    print("[2/2] Checking sample data...")
    if SAMPLE_DATA_PATH.exists():
        size_mb = SAMPLE_DATA_PATH.stat().st_size / (1024 * 1024)
        print(f"✓ Sample data exists: {SAMPLE_DATA_PATH.name} ({size_mb:.2f} MB)")
    else:
        print(f"❌ Sample data not found: {SAMPLE_DATA_PATH}")
        print(f"   Sample data should be included in the repository")
        print(f"   Expected location: {SAMPLE_DATA_PATH}")
        all_present = False

    print()

    # Summary
    print("="*60)
    if all_present:
        print("✓ All required files are present!")
        print("="*60)
        print("\nYou can now run inference:")
        print("    python run_inference.py")
    else:
        print("❌ Some files are missing")
        print("="*60)
        print("\nPlease resolve the missing files before running inference.")
        sys.exit(1)


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


if __name__ == "__main__":
    # Check if on Colab with Drive mounted
    if Path("/content").exists():
        check_google_drive_mounted()

    # Check and copy files
    check_and_copy_files()
