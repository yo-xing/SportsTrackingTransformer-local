#!/usr/bin/env python3
"""
Check for required files and show copy commands to sync from Google Drive.
"""
from pathlib import Path

def check_files():
    """Check for required files and show what's missing."""

    # Google Drive paths
    INPUT_DRIVE_DIR = Path("/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_gamestate")
    OUTPUT_DRIVE_DIR = Path("/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_7feat")

    # Local paths
    INPUT_DIR = Path("data/datasets_extra_gamestate_23")
    OUTPUT_DIR = Path("data/datasets_extra")

    splits = ["train", "val", "test"]
    model_type = "transformer"

    print("="*70)
    print("FILE STATUS CHECK")
    print("="*70)
    print()

    # Check source datasets (11 features)
    print("SOURCE DATASETS (11 features) - needed for filtering:")
    print("-" * 70)
    source_missing = []
    for split in splits:
        local_path = INPUT_DIR / model_type / f"{split}_dataset.pkl"
        drive_path = INPUT_DRIVE_DIR / model_type / f"{split}_dataset.pkl"

        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"✓ LOCAL:  {local_path}")
            print(f"          Size: {size_mb:.1f} MB")
        else:
            print(f"✗ MISSING: {local_path}")
            print(f"  DRIVE:   {drive_path}")
            source_missing.append((drive_path, local_path))
        print()

    # Check filtered datasets (7 features)
    print()
    print("FILTERED DATASETS (7 features) - needed for training:")
    print("-" * 70)
    filtered_missing = []
    for split in splits:
        local_path = OUTPUT_DIR / model_type / f"{split}_dataset.pkl"
        drive_path = OUTPUT_DRIVE_DIR / model_type / f"{split}_dataset.pkl"

        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"✓ LOCAL:  {local_path}")
            print(f"          Size: {size_mb:.1f} MB")
        else:
            print(f"✗ MISSING: {local_path}")
            print(f"  DRIVE:   {drive_path}")
            filtered_missing.append((drive_path, local_path))
        print()

    # Show copy commands
    print()
    print("="*70)
    print("COPY COMMANDS")
    print("="*70)
    print()

    if source_missing:
        print("# Copy source datasets (11 features) from Google Drive:")
        for drive_path, local_path in source_missing:
            print(f"mkdir -p {local_path.parent}")
            print(f"cp \"{drive_path}\" \"{local_path}\"")
            print()
    else:
        print("✓ All source datasets present locally")
        print()

    if filtered_missing:
        print("# Copy filtered datasets (7 features) from Google Drive:")
        for drive_path, local_path in filtered_missing:
            print(f"mkdir -p {local_path.parent}")
            print(f"cp \"{drive_path}\" \"{local_path}\"")
            print()
    else:
        print("✓ All filtered datasets present locally")
        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    total_missing = len(source_missing) + len(filtered_missing)
    if total_missing == 0:
        print("✓ All required files are present locally")
    else:
        print(f"⚠️  {len(source_missing)} source dataset(s) missing")
        print(f"⚠️  {len(filtered_missing)} filtered dataset(s) missing")
        print(f"\nTotal: {total_missing} file(s) need to be copied from Google Drive")
    print()

if __name__ == "__main__":
    check_files()
