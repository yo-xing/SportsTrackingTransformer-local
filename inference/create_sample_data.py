#!/usr/bin/env python3
"""
Create Sample Data for Inference

This script processes the raw sample data in inference/sample_data/Week 01/
and creates the preprocessed files needed for inference.

Unlike src/prep_extra_data.py which processes all data, this script only
processes the single sample file to keep inference fast and self-contained.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prep_extra_data import (
    map_column_names,
    filter_tracking_data,
    identify_ball_carrier,
    add_derived_features,
    determine_play_direction,
    standardize_tracking_directions,
    augment_mirror_tracking,
    add_relative_positions,
    get_yards_gained_target_df
)
import polars as pl


def process_sample_data(sample_data_dir: Path, output_dir: Path):
    """Process sample data from inference folder only."""

    print(f"\nProcessing sample data from: {sample_data_dir}")

    # Load only the RAW sample parquet file(s) from Week 01
    # Exclude already-processed files (inference_features.parquet, inference_targets.parquet)
    week_dir = sample_data_dir / "Week 01"
    if week_dir.exists():
        parquet_files = [f for f in week_dir.glob("*.parquet")
                        if not f.name.startswith("inference_")]
    else:
        parquet_files = [f for f in sample_data_dir.glob("*.parquet")
                        if not f.name.startswith("inference_")]

    if not parquet_files:
        print(f"❌ Error: No raw parquet files found in {sample_data_dir}")
        print(f"   Looking for files in: {sample_data_dir / 'Week 01'}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} raw parquet file(s):")
    for f in parquet_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")

    # Load data
    print("\nLoading data...")
    dfs = []
    for f in parquet_files:
        df = pl.read_parquet(f)
        dfs.append(df)

    df = pl.concat(dfs)
    print(f"Loaded {len(df):,} rows")

    # Apply preprocessing pipeline
    print("\nMapping column names...")
    df = map_column_names(df)

    print("Filtering tracking data...")
    df = filter_tracking_data(df)

    print("Identifying ball carriers...")
    df = identify_ball_carrier(df)

    print("Adding derived features...")
    df = add_derived_features(df)

    print("Determining play direction...")
    df = determine_play_direction(df)

    print("Standardizing tracking directions...")
    df = standardize_tracking_directions(df)

    print("Augmenting with mirrored plays...")
    df = augment_mirror_tracking(df)

    print("Adding relative positions...")
    df = add_relative_positions(df)

    print("Creating target labels...")
    df, targets_df = get_yards_gained_target_df(df)

    # Debug: Check if nflId column exists
    print(f"\nTotal columns after preprocessing: {len(df.columns)}")
    print(f"All columns: {df.columns}")
    if "nflId" not in df.columns:
        print("❌ WARNING: nflId column is missing!")
    else:
        print("✓ nflId column present")

    # Create train/val/test splits (70/15/15)
    print("\nCreating train/val/test splits...")
    unique_plays = df.select(["gameId", "playId"]).unique()
    n_plays = len(unique_plays)

    print(f"Total plays: {n_plays}")

    # Shuffle and split
    import random
    random.seed(42)
    play_indices = list(range(n_plays))
    random.shuffle(play_indices)

    n_train = int(0.7 * n_plays)
    n_val = int(0.15 * n_plays)

    train_indices = set(play_indices[:n_train])
    val_indices = set(play_indices[n_train:n_train + n_val])
    test_indices = set(play_indices[n_train + n_val:])

    # Assign split labels
    play_to_split = {}
    for idx, (game_id, play_id) in enumerate(unique_plays.iter_rows()):
        if idx in train_indices:
            play_to_split[(game_id, play_id)] = "train"
        elif idx in val_indices:
            play_to_split[(game_id, play_id)] = "val"
        else:
            play_to_split[(game_id, play_id)] = "test"

    # Add split column
    splits = []
    for game_id, play_id in zip(df["gameId"], df["playId"]):
        splits.append(play_to_split[(game_id, play_id)])

    df = df.with_columns(pl.Series("split", splits))

    # Save features and targets by split
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving preprocessed data to: {output_dir}")

    for split in ["train", "val", "test"]:
        split_df = df.filter(pl.col("split") == split)
        split_targets = targets_df.filter(
            pl.struct(["gameId", "playId"]).is_in(
                split_df.select(["gameId", "playId"]).unique()
            )
        )

        n_frames = len(split_df)
        n_plays = len(split_df.select(["gameId", "playId"]).unique())

        print(f"  {split}: {n_frames:,} frames, {n_plays} plays")

        # Save features
        features_file = output_dir / f"{split}_features.parquet"
        split_df.write_parquet(features_file)

        # Save targets
        targets_file = output_dir / f"{split}_targets.parquet"
        split_targets.write_parquet(targets_file)

    print("\n✓ Sample data preprocessing complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process sample data for inference")
    parser.add_argument("--data-file", type=str, default=None,
                       help="Path to specific parquet file to process")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "split_prepped_data_extra"

    # Determine data source
    if args.data_file:
        # Use specific file provided
        data_file = Path(args.data_file)
        if not data_file.exists():
            print(f"❌ Error: Data file not found: {data_file}")
            sys.exit(1)
        sample_data_dir = data_file.parent
        print(f"Using custom data file: {data_file}")
    else:
        # Use default sample_data directory
        sample_data_dir = script_dir / "sample_data"
        if not sample_data_dir.exists():
            print(f"❌ Error: Sample data directory not found: {sample_data_dir}")
            sys.exit(1)

    # Process sample data
    process_sample_data(sample_data_dir, output_dir)
