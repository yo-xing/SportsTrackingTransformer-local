#!/usr/bin/env python3
"""
Sports Tracking Transformer - Simple Inference Script

This script runs inference on sample data using a trained model checkpoint.
It supports both preprocessed sample data and raw Axially-format data.

Usage:
    python run_inference.py

Requirements:
    - Model checkpoint in inference/model/
    - Sample data in inference/sample_data/

Output:
    - predictions.csv with predicted yards for each frame
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import torch
import pandas as pd
import numpy as np
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from models import LitModel
from datasets import BDB2024_Dataset, NUM_YARDS_CLASSES, MIN_YARDS


# Paths
INFERENCE_DIR = Path(__file__).parent
SAMPLE_DATA_DIR = INFERENCE_DIR / "sample_data"
MODEL_DIR = INFERENCE_DIR / "model"


def find_checkpoint():
    """
    Find the model checkpoint file in the model/ directory.

    Returns:
        Path to checkpoint file

    Raises:
        FileNotFoundError if no checkpoint found
    """
    ckpt_files = list(MODEL_DIR.glob("*.ckpt"))

    if not ckpt_files:
        raise FileNotFoundError(
            f"No .ckpt files found in {MODEL_DIR}/\n"
            f"Please place a model checkpoint in the model/ directory."
        )

    if len(ckpt_files) > 1:
        print(f"Found {len(ckpt_files)} checkpoint files. Using the first one:")
        for f in ckpt_files:
            print(f"  - {f.name}")
        print()

    return ckpt_files[0]


def load_model(checkpoint_path):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to .ckpt file

    Returns:
        Tuple of (model, device)
    """
    print(f"Loading model from: {checkpoint_path.name}")

    # Load checkpoint
    model = LitModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU detected, using CPU (this may be slower)")

    model = model.to(device)

    # Print model info
    print(f"\nModel Configuration:")
    print(f"  Architecture: {model.hparams.get('model_type', 'unknown')}")
    print(f"  Model Dimension: {model.hparams.get('model_dim', 'unknown')}")
    print(f"  Layers: {model.hparams.get('num_layers', 'unknown')}")
    print(f"  Learning Rate: {model.hparams.get('lr', 'unknown')}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    return model, device


def check_sample_data():
    """
    Check what sample data is available.

    Returns:
        str: "preprocessed" if _sample.parquet files exist,
             "raw" if Week 01/ directory exists,
             None if no data found
    """
    # Check for preprocessed samples
    train_sample = SAMPLE_DATA_DIR / "train_features_sample.parquet"
    if train_sample.exists():
        print(f"Found preprocessed sample data:")
        for split in ["train", "val", "test"]:
            features_path = SAMPLE_DATA_DIR / f"{split}_features_sample.parquet"
            targets_path = SAMPLE_DATA_DIR / f"{split}_targets_sample.parquet"
            if features_path.exists() and targets_path.exists():
                features_df = pl.read_parquet(features_path)
                targets_df = pl.read_parquet(targets_path)
                print(f"  {split}: {len(features_df)} feature rows, {len(targets_df)} target rows")
        return "preprocessed"

    # Check for raw data
    raw_data_dir = SAMPLE_DATA_DIR / "Week 01"
    if raw_data_dir.exists():
        parquet_files = list(raw_data_dir.glob("*.parquet"))
        if parquet_files:
            print(f"Found raw Axially-format data:")
            for f in parquet_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name} ({size_mb:.1f} MB)")
            return "raw"

    return None


def preprocess_raw_data():
    """
    Preprocess raw Axially-format data from sample_data/Week 01/

    This implements the preprocessing pipeline from src/prep_extra_data.py
    adapted to work with local paths only (no Google Drive dependencies).

    Returns:
        bool: True if preprocessing succeeded, False otherwise
    """
    print("\nPreprocessing raw Axially-format data...")
    print("This may take a few minutes...")

    try:
        # Step 1: Load raw data
        print("\n[1/11] Loading raw data...")
        raw_data_dir = SAMPLE_DATA_DIR / "Week 01"
        parquet_files = list(raw_data_dir.glob("*.parquet"))

        if not parquet_files:
            print(f"ERROR: No parquet files found in {raw_data_dir}")
            return False

        print(f"Found {len(parquet_files)} files")
        dfs = []
        for f in parquet_files:
            df = pl.read_parquet(f)
            dfs.append(df)

        df = pl.concat(dfs, how="vertical_relaxed")
        print(f"Loaded {len(df)} rows")

        # Step 2: Map column names
        print("\n[2/11] Mapping column names (snake_case → camelCase)...")
        df = df.rename({
            "gamekey": "gameId",
            "playid": "playId",
            "nfl_id": "nflId",
            "frame_id": "frameId",
            "display_name": "displayName",
            "team_abbr": "club",
            "yards_to_go": "yardsToGo",
            "yards_gained": "prePenaltyPlayResult",
            "vel_x": "vx",
            "vel_y": "vy",
            "vel": "s",
            "roster_position": "position",
        })
        print("Column names mapped")

        # Step 3: Filter tracking data
        print("\n[3/11] Filtering tracking data...")
        og_len = len(df)
        # Remove football rows
        df = df.filter(pl.col("possession_status") != "ball")
        print(f"Removed football rows: {og_len - len(df)} rows")

        # Keep only pass plays
        og_len = len(df)
        df = df.filter(pl.col("play_type") == "play_type_pass")
        print(f"Filtered to pass plays: {og_len - len(df)} rows removed")

        # Step 4: Identify ball carrier
        print("\n[4/11] Identifying ball carriers...")
        # Load raw data again to get football positions
        raw_df = pl.concat([pl.read_parquet(f) for f in parquet_files], how="vertical_relaxed")
        raw_df = raw_df.rename({
            "gamekey": "gameId",
            "playid": "playId",
            "nfl_id": "nflId",
            "frame_id": "frameId",
        })
        raw_df = raw_df.filter(pl.col("play_type") == "play_type_pass")

        # Get football positions
        football_df = raw_df.filter(pl.col("possession_status") == "ball")

        # Find ball carrier for each play
        ball_carriers = []
        priority_events = ["handoff", "first_contact", "ball_snap"]

        for (game_id, play_id), group in df.group_by(["gameId", "playId"]):
            play_football = football_df.filter(
                (pl.col("gameId") == game_id) & (pl.col("playId") == play_id)
            )

            if len(play_football) == 0:
                continue

            # Find event frame
            event_frame = None
            for event in priority_events:
                event_rows = play_football.filter(pl.col("event") == event)
                if len(event_rows) > 0:
                    event_frame = event_rows["frameId"].min()
                    break

            if event_frame is None:
                event_frame = play_football["frameId"].median()

            # Get football position at event frame
            fb_at_event = play_football.filter(pl.col("frameId") == event_frame)
            if len(fb_at_event) == 0:
                fb_at_event = play_football.filter(
                    pl.col("frameId") == play_football["frameId"].min()
                )

            fb_x = fb_at_event["x"].item() if len(fb_at_event) > 0 else None
            fb_y = fb_at_event["y"].item() if len(fb_at_event) > 0 else None

            if fb_x is None or fb_y is None:
                continue

            # Get offensive players at event frame
            off_players = group.filter(
                (pl.col("frameId") == event_frame) &
                (pl.col("possession_status") == "off")
            )

            if len(off_players) == 0:
                continue

            # Find closest player
            off_players = off_players.with_columns(
                dist=((pl.col("x") - fb_x) ** 2 + (pl.col("y") - fb_y) ** 2).sqrt()
            )

            closest = off_players.sort("dist").head(1)
            if len(closest) > 0:
                ball_carriers.append({
                    "gameId": game_id,
                    "playId": play_id,
                    "ballCarrierId": closest["nflId"].item(),
                })

        ball_carrier_df = pl.DataFrame(ball_carriers)
        print(f"Identified ball carriers for {len(ball_carrier_df)} plays")
        df = df.join(ball_carrier_df, on=["gameId", "playId"], how="inner")

        # Step 5: Add derived features
        print("\n[5/11] Adding derived features...")
        df = df.with_columns(
            is_ball_carrier=(pl.col("nflId") == pl.col("ballCarrierId")).cast(int),
            side=pl.when(pl.col("possession_status") == "off")
            .then(pl.lit(1))
            .otherwise(pl.lit(-1)),
            weight_Z=pl.lit(0.0),
            height_Z=pl.lit(0.0),
        )

        # Convert orientation to unit vectors
        df = df.with_columns(
            o_rad=((pl.col("o") - 90) * -1) % 360,
        ).with_columns(
            ox=pl.col("o_rad").radians().cos(),
            oy=pl.col("o_rad").radians().sin(),
        ).drop("o_rad")

        # Calculate distance to goal
        df = df.with_columns(
            distanceToGoal=(100 - pl.col("line_of_scrimmage")).cast(pl.Float64)
        )
        print("Added derived features")

        # Step 6: Determine play direction
        print("\n[6/11] Determining play directions...")
        play_directions = (
            df.filter(pl.col("side") == 1)
            .group_by(["gameId", "playId"])
            .agg([
                pl.col("x").filter(pl.col("frameId") == pl.col("frameId").min()).mean().alias("avg_off_x"),
                pl.col("line_of_scrimmage").first().alias("los"),
            ])
            .with_columns(
                playDirection=pl.when(pl.col("avg_off_x") > pl.col("los"))
                .then(pl.lit("left"))
                .otherwise(pl.lit("right"))
            )
            .select(["gameId", "playId", "playDirection"])
        )
        df = df.join(play_directions, on=["gameId", "playId"], how="left")
        print(f"Determined directions for {len(play_directions)} plays")

        # Step 7: Standardize tracking directions
        print("\n[7/11] Standardizing tracking directions...")
        df = df.with_columns(
            x=pl.when(pl.col("playDirection") == "right").then(pl.col("x")).otherwise(120 - pl.col("x")),
            y=pl.when(pl.col("playDirection") == "right").then(pl.col("y")).otherwise(53.3 - pl.col("y")),
            vx=pl.when(pl.col("playDirection") == "right").then(pl.col("vx")).otherwise(-1 * pl.col("vx")),
            vy=pl.when(pl.col("playDirection") == "right").then(pl.col("vy")).otherwise(-1 * pl.col("vy")),
        )
        print("Standardized directions (all plays left-to-right)")

        # Step 8: Augment with mirrored data (skip for inference - we don't need augmentation)
        print("\n[8/11] Skipping data augmentation (not needed for inference)...")
        df = df.with_columns(mirrored=pl.lit(False))

        # Step 9: Add relative positions
        print("\n[9/11] Adding relative positions...")
        # Get ball carrier positions per frame
        bc_positions = (
            df.filter(pl.col("is_ball_carrier") == 1)
            .select(["gameId", "playId", "frameId", "x", "y", "vx", "vy"])
            .rename({"x": "bc_x", "y": "bc_y", "vx": "bc_vx", "vy": "bc_vy"})
        )

        # Join and calculate relative positions
        df = df.join(bc_positions, on=["gameId", "playId", "frameId"], how="left")
        df = df.with_columns(
            x_rel=(pl.col("x") - pl.col("bc_x")).cast(pl.Float64),
            y_rel=(pl.col("y") - pl.col("bc_y")).cast(pl.Float64),
        )
        print("Added relative positions")

        # Step 10: Generate targets
        print("\n[10/11] Generating targets...")
        target_df = (
            df.group_by(["gameId", "playId", "frameId", "mirrored"])
            .agg([
                pl.col("prePenaltyPlayResult").first().alias("yards_gained"),
            ])
            .with_columns(
                yards_gained_class=(pl.col("yards_gained") - pl.lit(-10)).cast(int).clip(0, 109)
            )
        )
        print(f"Generated {len(target_df)} target rows")

        # Step 11: Save preprocessed data
        print("\n[11/11] Saving preprocessed data...")
        OUTPUT_DIR = SAMPLE_DATA_DIR / "processed"
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

        # For inference, we don't split - just save all as one file
        # Select only the features we need
        features_df = df.select([
            "gameId", "playId", "frameId", "nflId",
            "x_rel", "y_rel", "vx", "vy", "side", "is_ball_carrier",
            "distanceToGoal", "mirrored"
        ])

        features_path = OUTPUT_DIR / "inference_features.parquet"
        targets_path = OUTPUT_DIR / "inference_targets.parquet"

        features_df.write_parquet(features_path)
        target_df.write_parquet(targets_path)

        print(f"Saved: {features_path}")
        print(f"Saved: {targets_path}")
        print(f"\nPreprocessing complete!")
        print(f"  Features: {len(features_df)} rows")
        print(f"  Targets: {len(target_df)} rows")

        return True

    except Exception as e:
        print(f"\nERROR during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_dataset_from_files(split, model_type):
    """
    Load dataset directly from parquet files without using datasets.py's load_datasets()
    to avoid Google Drive dependencies.

    Args:
        split: "train", "val", or "test" (ignored if using inference files)
        model_type: "transformer" or "zoo"

    Returns:
        BDB2024_Dataset instance
    """
    # Try preprocessed inference files first (from preprocess_raw_data)
    inference_features_path = SAMPLE_DATA_DIR / "processed" / "inference_features.parquet"
    inference_targets_path = SAMPLE_DATA_DIR / "processed" / "inference_targets.parquet"

    if inference_features_path.exists() and inference_targets_path.exists():
        print(f"  Loading from preprocessed inference files...")
        features_df = pl.read_parquet(inference_features_path)
        targets_df = pl.read_parquet(inference_targets_path)
    else:
        # Try split-specific sample files
        sample_features_path = SAMPLE_DATA_DIR / f"{split}_features_sample.parquet"
        sample_targets_path = SAMPLE_DATA_DIR / f"{split}_targets_sample.parquet"

        if sample_features_path.exists() and sample_targets_path.exists():
            print(f"  Loading from sample files...")
            features_df = pl.read_parquet(sample_features_path)
            targets_df = pl.read_parquet(sample_targets_path)
        else:
            # Fall back to full preprocessed files
            full_features_path = Path("data/split_prepped_data_extra") / f"{split}_features.parquet"
            full_targets_path = Path("data/split_prepped_data_extra") / f"{split}_targets.parquet"

            if not (full_features_path.exists() and full_targets_path.exists()):
                raise FileNotFoundError(
                    f"Could not find data for {split} split.\n"
                    f"Looked for:\n"
                    f"  - {inference_features_path}\n"
                    f"  - {sample_features_path}\n"
                    f"  - {full_features_path}\n"
                    f"\nPlease run preprocessing first or provide preprocessed data"
                )

            print(f"  Loading from full preprocessed files...")
            features_df = pl.read_parquet(full_features_path)
            targets_df = pl.read_parquet(full_targets_path)

            # Filter to non-mirrored data only
            print(f"  Filtering to mirrored=False...")
            features_df = features_df.filter(pl.col("mirrored") == False)
            targets_df = targets_df.filter(pl.col("mirrored") == False)

    # Create dataset
    # Note: BDB2024_Dataset expects data in a specific format
    # We'll create a simple wrapper that holds the data

    class SimpleDataset:
        """Simple dataset wrapper for preprocessed data."""

        def __init__(self, features_df, targets_df, model_type):
            self.features_df = features_df
            self.targets_df = targets_df
            self.model_type = model_type

            # Convert to pandas for easier indexing
            self.features_pd = features_df.to_pandas()
            self.targets_pd = targets_df.to_pandas()

            # Group by (gameId, playId, frameId) for efficient lookups
            self.features_pd = self.features_pd.set_index(["gameId", "playId", "frameId"])
            self.targets_pd = self.targets_pd.set_index(["gameId", "playId", "frameId"])

            # Get unique frames
            self.frames = list(self.targets_pd.index.unique())

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, idx):
            """
            Get a single frame of data.

            Returns:
                Tuple of (features, target)
                - features: tensor of shape (23, 7) for 23 entities × 7 features
                - target: scalar yards_gained_class (0-109)
            """
            game_id, play_id, frame_id = self.frames[idx]

            # Get features for this frame
            frame_features = self.features_pd.loc[(game_id, play_id, frame_id)]

            # Handle case where there might be multiple rows (shouldn't happen after proper preprocessing)
            if isinstance(frame_features, pd.DataFrame):
                frame_features = frame_features.iloc[0]

            # Extract feature columns
            # Expected: x_rel, y_rel, vx, vy, side, is_ball_carrier, distanceToGoal
            feature_cols = ["x_rel", "y_rel", "vx", "vy", "side", "is_ball_carrier"]

            # Get all 23 entities for this frame
            entities_features = self.features_pd.loc[(game_id, play_id, frame_id), feature_cols]

            # If single row, need to reshape
            if not isinstance(entities_features, pd.DataFrame):
                entities_features = pd.DataFrame([entities_features])

            # Get distanceToGoal (should be same for all entities in frame)
            distance_to_goal = self.features_pd.loc[(game_id, play_id, frame_id), "distanceToGoal"]
            if isinstance(distance_to_goal, pd.Series):
                distance_to_goal = distance_to_goal.iloc[0]

            # Combine entity features + game state feature
            # Shape: (23, 7) where 7 = 6 entity features + 1 game state feature
            entity_features_np = entities_features.values  # (N, 6)

            # Pad or truncate to exactly 23 entities
            if len(entity_features_np) < 23:
                # Pad with zeros
                padding = np.zeros((23 - len(entity_features_np), 6))
                entity_features_np = np.vstack([entity_features_np, padding])
            elif len(entity_features_np) > 23:
                # Truncate (shouldn't happen)
                entity_features_np = entity_features_np[:23]

            # Add distance_to_goal as 7th feature column
            distance_col = np.full((23, 1), distance_to_goal)
            features_np = np.hstack([entity_features_np, distance_col])  # (23, 7)

            features_tensor = torch.FloatTensor(features_np)

            # Get target
            target_row = self.targets_pd.loc[(game_id, play_id, frame_id)]
            if isinstance(target_row, pd.DataFrame):
                target_row = target_row.iloc[0]

            yards_class = int(target_row["yards_gained_class"])
            target_tensor = torch.LongTensor([yards_class])

            return features_tensor, target_tensor, (game_id, play_id, frame_id)

    dataset = SimpleDataset(features_df, targets_df, model_type)
    print(f"  Created dataset with {len(dataset)} frames")

    return dataset


def run_inference(model, dataset, device, split):
    """
    Run inference on a dataset.

    Args:
        model: Trained model
        dataset: Dataset to run inference on
        device: torch device
        split: "train", "val", or "test"

    Returns:
        DataFrame with predictions
    """
    model.eval()

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    all_predictions = []
    all_targets = []
    all_keys = []

    print(f"  Running inference on {len(dataset)} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            features, targets, keys = batch

            features = features.to(device)

            # Forward pass
            logits = model(features)  # (batch_size, 110)

            # Convert to probabilities
            probs = torch.softmax(logits, dim=1)

            # Get predicted class and expected yards
            predicted_classes = torch.argmax(logits, dim=1)

            # Calculate expected yards from probability distribution
            yard_values = torch.arange(MIN_YARDS, MIN_YARDS + NUM_YARDS_CLASSES).to(device)
            expected_yards = (probs * yard_values).sum(dim=1)

            # Get confidence (max probability)
            confidences = probs.max(dim=1).values

            all_predictions.append({
                'predicted_classes': predicted_classes.cpu().numpy(),
                'expected_yards': expected_yards.cpu().numpy(),
                'confidences': confidences.cpu().numpy(),
            })
            all_targets.append(targets.cpu().numpy())
            all_keys.extend(keys)

            if (batch_idx + 1) % 10 == 0:
                print(f"    Processed {(batch_idx + 1) * len(features)} / {len(dataset)} samples...")

    # Combine all predictions
    predicted_classes = np.concatenate([p['predicted_classes'] for p in all_predictions])
    expected_yards = np.concatenate([p['expected_yards'] for p in all_predictions])
    confidences = np.concatenate([p['confidences'] for p in all_predictions])
    targets = np.concatenate(all_targets)

    # Convert predicted classes to yards
    predicted_class_yards = predicted_classes + MIN_YARDS

    # Convert target classes to yards
    actual_yards = targets.flatten() + MIN_YARDS

    # Create results DataFrame
    results = []
    for i, (game_id, play_id, frame_id) in enumerate(all_keys):
        results.append({
            'dataset_split': split,
            'gameId': game_id,
            'playId': play_id,
            'frameId': frame_id,
            'predicted_yards': expected_yards[i],
            'predicted_class_yards': predicted_class_yards[i],
            'confidence': confidences[i],
            'actual_yards': actual_yards[i],
        })

    df = pd.DataFrame(results)

    # Calculate metrics
    mae = np.abs(df['predicted_yards'] - df['actual_yards']).mean()
    num_frames = len(df)
    num_plays = df.groupby(['gameId', 'playId']).ngroups
    avg_frames_per_play = num_frames / num_plays if num_plays > 0 else 0

    print(f"\n  {split.upper()} Metrics:")
    print(f"    Frames: {num_frames}")
    print(f"    Plays: {num_plays}")
    print(f"    Avg frames/play: {avg_frames_per_play:.1f}")
    print(f"    MAE: {mae:.2f} yards")
    print(f"    Mean predicted: {df['predicted_yards'].mean():.2f} yards")
    print(f"    Mean actual: {df['actual_yards'].mean():.2f} yards")

    return df


def collate_fn(batch):
    """Custom collate function for dataloader."""
    features = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    keys = [item[2] for item in batch]
    return features, targets, keys


def main():
    """Main inference pipeline."""
    print("="*60)
    print("Sports Tracking Transformer - Simple Inference")
    print("="*60)
    print()

    # Step 1: Check sample data
    print("[1/4] Checking sample data...")
    data_type = check_sample_data()

    if data_type is None:
        print("\nERROR: No sample data found!")
        print("Please place data in one of these locations:")
        print("  - inference/sample_data/*_features_sample.parquet (preprocessed)")
        print("  - inference/sample_data/Week 01/*.parquet (raw)")
        return

    if data_type == "raw":
        print("\nPreprocessing raw data...")
        success = preprocess_raw_data()
        if not success:
            return

    print()

    # Step 2: Load model
    print("[2/4] Loading model...")
    try:
        checkpoint_path = find_checkpoint()
        model, device = load_model(checkpoint_path)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Step 3: Run inference
    all_predictions = []
    model_type = model.hparams.get('model_type', 'transformer')

    # Check if we have preprocessed inference files (single file, no splits)
    inference_features_path = SAMPLE_DATA_DIR / "processed" / "inference_features.parquet"

    if inference_features_path.exists():
        print(f"[3/4] Running inference on preprocessed data...")

        try:
            # Load dataset
            dataset = load_dataset_from_files("inference", model_type)

            # Run inference
            predictions_df = run_inference(model, dataset, device, split="inference")
            all_predictions.append(predictions_df)

        except Exception as e:
            print(f"  ERROR processing inference data: {e}")
            import traceback
            traceback.print_exc()
            return

        print()
    else:
        # We have split data (train/val/test)
        for idx, split in enumerate(["train", "val", "test"], start=3):
            print(f"[{idx}/5] Running inference on {split} set...")

            try:
                # Load dataset
                dataset = load_dataset_from_files(split, model_type)

                # Run inference
                predictions_df = run_inference(model, dataset, device, split=split)
                all_predictions.append(predictions_df)

            except FileNotFoundError as e:
                print(f"  WARNING: Skipping {split} - {e}")
                continue
            except Exception as e:
                print(f"  ERROR processing {split}: {e}")
                import traceback
                traceback.print_exc()
                continue

            print()

    if not all_predictions:
        print("\nERROR: No predictions were generated!")
        return

    # Step 4: Combine and save results
    print("[4/4] Saving predictions...")
    combined_predictions = pd.concat(all_predictions, ignore_index=True)

    output_path = INFERENCE_DIR / "predictions.csv"
    combined_predictions.to_csv(output_path, index=False)

    print(f"  Saved to: {output_path}")
    print()

    # Print summary
    print("="*60)
    print("Inference Complete!")
    print("="*60)

    # Overall statistics
    total_frames = len(combined_predictions)
    total_plays = combined_predictions.groupby(['gameId', 'playId']).ngroups
    print(f"Total predictions: {total_frames} frames across {total_plays} plays")
    print()

    # Per-split statistics
    for split in ["train", "val", "test", "inference"]:
        split_df = combined_predictions[combined_predictions['dataset_split'] == split]
        if len(split_df) > 0:
            mae = np.abs(split_df['predicted_yards'] - split_df['actual_yards']).mean()
            num_frames = len(split_df)
            num_plays = split_df.groupby(['gameId', 'playId']).ngroups
            avg_frames_per_play = num_frames / num_plays

            print(f"  {split.upper()}:")
            print(f"    Frames: {num_frames}")
            print(f"    Plays: {num_plays}")
            print(f"    Avg frames/play: {avg_frames_per_play:.1f}")
            print(f"    MAE: {mae:.2f} yards")
            print()

    print(f"Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
