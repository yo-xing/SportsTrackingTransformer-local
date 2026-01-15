"""
Script to filter precomputed dataset features.

This script loads precomputed datasets that have 11 features and filters them
to only keep 7 features (removing yardsToGo, down, quarter, half_seconds_remaining).

The 11 original features are:
[0] x_rel, [1] y_rel, [2] vx, [3] vy, [4] side, [5] is_ball_carrier,
[6] yardsToGo, [7] down, [8] distanceToGoal, [9] quarter, [10] half_seconds_remaining

We keep indices [0, 1, 2, 3, 4, 5, 8] = 7 features total
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Indices to keep from the original 11 features
KEEP_INDICES = [0, 1, 2, 3, 4, 5, 8]  # x_rel, y_rel, vx, vy, side, is_ball_carrier, distanceToGoal

# Input: 11-feature datasets from add-game-state-features branch
INPUT_DIR = Path("data/datasets_extra_gamestate_23/")
# Output: 7-feature datasets for 23-entity branch
OUTPUT_DIR = Path("data/datasets_extra/")


class _DatasetUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps __main__.BDB2024_Dataset to datasets.BDB2024_Dataset."""
    def find_class(self, module, name):
        if module == "__main__" and name == "BDB2024_Dataset":
            from datasets import BDB2024_Dataset
            return BDB2024_Dataset
        return super().find_class(module, name)


def filter_dataset(input_path: Path, output_path: Path, model_type: str):
    """
    Load a precomputed dataset and filter its features.

    Args:
        input_path: Path to input dataset pickle file
        output_path: Path to output filtered dataset pickle file
        model_type: Type of model ('transformer' or 'zoo')
    """
    print(f"Loading dataset from {input_path}...")
    with open(input_path, "rb") as f:
        dataset = _DatasetUnpickler(f).load()

    if model_type == "transformer":
        print(f"Filtering features from 11 to 7...")
        # Filter the feature arrays in the dataset
        for key in tqdm(dataset.feature_arrays.keys(), desc="Filtering features"):
            original_array = dataset.feature_arrays[key]
            # original_array shape: (22, 11) for 22 players, 11 features
            # new shape should be: (22, 7)
            filtered_array = original_array[:, KEEP_INDICES]
            dataset.feature_arrays[key] = filtered_array

        # Update feature_len if it exists as an attribute
        print(f"Original feature array shape example: {original_array.shape}")
        print(f"Filtered feature array shape example: {filtered_array.shape}")
    else:
        print(f"Zoo model - no feature filtering needed, copying as-is...")

    # Save the filtered dataset
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving filtered dataset to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Done! Filtered dataset saved to {output_path}")


def main(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR):
    """Filter all datasets in the input directory."""
    for model_type in ["transformer", "zoo"]:
        model_dir = input_dir / model_type
        if not model_dir.exists():
            print(f"Warning: {model_dir} does not exist, skipping...")
            continue

        for split in ["train", "val", "test"]:
            input_file = model_dir / f"{split}_dataset.pkl"
            if not input_file.exists():
                print(f"Warning: {input_file} does not exist, skipping...")
                continue

            output_file = output_dir / model_type / f"{split}_dataset.pkl"

            print(f"\n{'='*60}")
            print(f"Processing {model_type}/{split}")
            print(f"{'='*60}")

            filter_dataset(input_file, output_file, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter precomputed dataset features")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Input directory containing datasets to filter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for filtered datasets (default: overwrites input)",
    )
    args = parser.parse_args()

    main(input_dir=args.input_dir, output_dir=args.output_dir)
