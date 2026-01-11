"""
Dataset Module for NFL Big Data Bowl 2024

This module handles data loading and preprocessing for yards gained prediction models.
The target is a probability distribution over yards gained (-10 to +99 yards),
represented as a class index (0-109) for cross-entropy loss.
"""

import gc
import multiprocessing as mp
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import Dataset
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

PREPPED_DATA_DIR = Path("data/split_prepped_data/")
DATASET_DIR = Path("data/datasets/")

# Yards gained classification constants
# Class 0 = -10 yards, Class 109 = +99 yards
MIN_YARDS = -10
MAX_YARDS = 99
NUM_YARDS_CLASSES = MAX_YARDS - MIN_YARDS + 1  # 110 classes


def _malloc_trim():
    """Best-effort: return freed heap to OS on Linux/glibc (can help with fragmentation)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class BDB2024_Dataset(Dataset):
    """
    Custom dataset class for NFL tracking data.
    """

    def __init__(
        self,
        model_type: str,
        feature_df: pl.DataFrame,
        tgt_df: pl.DataFrame,
    ):
        if model_type not in ["transformer", "zoo"]:
            raise ValueError("model_type must be either 'transformer' or 'zoo'")

        self.model_type = model_type

        # Sort keys to ensure deterministic ordering across runs
        self.keys = sorted(feature_df.select(["gameId", "playId", "mirrored", "frameId"]).unique().rows())

        # Convert to pandas form with index for quick row retrieval
        self.feature_df_partition = (
            feature_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["gameId", "playId", "mirrored", "frameId", "nflId"])
            .sort_index()
        )
        self.tgt_df_partition = (
            tgt_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["gameId", "playId", "mirrored", "frameId"])
            .sort_index()
        )

        # Precompute features and store in dicts (keep same external behavior)
        self.tgt_arrays: dict[tuple, np.ndarray] = {}
        self.feature_arrays: dict[tuple, np.ndarray] = {}

        # ---- Minimal performance/memory tweaks ----
        n_workers = 6

        # Keep chunking to bound peak RAM. 50k worked, but you still OOM on train by chunk ~4.
        # This reduces peak further without meaningfully changing compute time.
        MAX_KEYS_PER_CHUNK = 35_000

        n = len(self.keys)
        n_chunks = (n + MAX_KEYS_PER_CHUNK - 1) // MAX_KEYS_PER_CHUNK

        with mp.Pool(processes=n_workers) as pool:
            for chunk_idx in range(n_chunks):
                start = chunk_idx * MAX_KEYS_PER_CHUNK
                end = min(start + MAX_KEYS_PER_CHUNK, n)
                keys_chunk = self.keys[start:end]
                if not keys_chunk:
                    continue

                # Larger mp chunksize => less IPC overhead (closer to pool.map speed)
                mp_chunksize = max(512, len(keys_chunk) // (n_workers * 2))

                it = pool.imap_unordered(self.process_key, keys_chunk, chunksize=mp_chunksize)

                # Reduce tqdm overhead (updating per-item can slow a lot)
                with tqdm(
                    total=len(keys_chunk),
                    desc=f"Pre-computing feature transforms (chunk {chunk_idx+1}/{n_chunks})",
                    dynamic_ncols=True,
                    mininterval=2.0,
                    smoothing=0.0,
                ) as pbar:
                    for key, tgt_array, feature_array in it:
                        self.tgt_arrays[key] = tgt_array
                        self.feature_arrays[key] = feature_array
                        pbar.update(1)

                # Help Python + glibc release/compact between chunks
                gc.collect()
                _malloc_trim()

        # Drop big pandas partitions before pickling dataset (reduces RAM spikes + output size)
        self.feature_df_partition = None
        self.tgt_df_partition = None
        gc.collect()
        _malloc_trim()

    def process_key(self, key: tuple) -> tuple[tuple, np.ndarray, np.ndarray]:
        tgt_array = self.transform_target_df(self.tgt_df_partition.loc[key])
        feature_array = self.transform_input_frame_df(self.feature_df_partition.loc[key])
        return key, tgt_array, feature_array

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        key = self.keys[idx]
        return self.feature_arrays[key], self.tgt_arrays[key]

    def transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        if self.model_type == "transformer":
            return self.transformer_transform_input_frame_df(frame_df)
        elif self.model_type == "zoo":
            return self.zoo_transform_input_frame_df(frame_df)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def transform_target_df(self, tgt_df: pd.DataFrame | pd.Series) -> np.ndarray:
        """Return yards_gained_class as scalar int64 for cross-entropy loss."""
        # When .loc[key] matches a single row, pandas returns a Series instead of DataFrame
        if isinstance(tgt_df, pd.Series):
            y = np.int64(tgt_df["yards_gained_class"])
        else:
            y = tgt_df["yards_gained_class"].to_numpy(dtype=np.int64).squeeze()
        return y

    def transformer_transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        features = ["x_rel", "y_rel", "vx", "vy", "side", "is_ball_carrier"]
        x = frame_df[features].to_numpy(dtype=np.float32)
        assert x.shape == (22, len(features)), f"Expected shape (22, {len(features)}), got {x.shape}"
        return x

    def zoo_transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        ball_carrier = frame_df[frame_df["is_ball_carrier"] == 1]
        off_plyrs = frame_df[(frame_df["side"] == 1) & (frame_df["is_ball_carrier"] == 0)]
        def_plyrs = frame_df[frame_df["side"] == -1]

        ball_carr_mvmt_feats = ball_carrier[["x_rel", "y_rel", "vx", "vy"]].to_numpy(dtype=np.float32).squeeze()
        off_mvmt_feats = off_plyrs[["x_rel", "y_rel", "vx", "vy"]].to_numpy(dtype=np.float32)
        def_mvmt_feats = def_plyrs[["x_rel", "y_rel", "vx", "vy"]].to_numpy(dtype=np.float32)

        x = [
            np.tile(def_mvmt_feats[:, 2:], (10, 1, 1)),
            np.tile(def_mvmt_feats[None, :, :2] - ball_carr_mvmt_feats[None, None, :2], (10, 1, 1)),
            np.tile(def_mvmt_feats[None, :, 2:] - ball_carr_mvmt_feats[None, None, 2:], (10, 1, 1)),
            off_mvmt_feats[:, None, :2] - def_mvmt_feats[None, :, :2],
            off_mvmt_feats[:, None, 2:] - def_mvmt_feats[None, :, 2:],
        ]

        x = np.concatenate(x, dtype=np.float32, axis=-1)
        assert x.shape == (10, 11, 10), f"Expected shape (10, 11, 10), got {x.shape}"
        return x


def load_datasets(model_type: str, split: str) -> BDB2024_Dataset:
    ds_dir = DATASET_DIR / model_type
    file_path = ds_dir / f"{split}_dataset.pkl"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, "rb") as f:
        return pickle.load(f)


def _read_features(split: str) -> pl.DataFrame:
    """
    Read ONLY the columns actually used downstream, and downcast to reduce RAM before pandas conversion.
    This is the biggest win for the train split.
    """
    path = PREPPED_DATA_DIR / f"{split}_features.parquet"
    cols = [
        "gameId",
        "playId",
        "mirrored",
        "frameId",
        "nflId",
        "x_rel",
        "y_rel",
        "vx",
        "vy",
        "side",
        "is_ball_carrier",
    ]
    df = pl.read_parquet(path, columns=cols)

    # Downcast: big memory win on train
    df = df.with_columns(
        [
            pl.col("x_rel").cast(pl.Float32),
            pl.col("y_rel").cast(pl.Float32),
            pl.col("vx").cast(pl.Float32),
            pl.col("vy").cast(pl.Float32),
            pl.col("side").cast(pl.Int8),
            pl.col("is_ball_carrier").cast(pl.Int8),
            pl.col("mirrored").cast(pl.Boolean),
            pl.col("gameId").cast(pl.Int32),
            pl.col("playId").cast(pl.Int32),
            pl.col("frameId").cast(pl.Int32),
            pl.col("nflId").cast(pl.Int32),
        ]
    )
    return df


def _read_targets(split: str) -> pl.DataFrame:
    """
    Read ONLY the target columns used, and downcast.
    """
    path = PREPPED_DATA_DIR / f"{split}_targets.parquet"
    cols = [
        "gameId",
        "playId",
        "mirrored",
        "frameId",
        "yards_gained_class",
    ]
    df = pl.read_parquet(path, columns=cols)

    df = df.with_columns(
        [
            pl.col("yards_gained_class").cast(pl.Int64),
            pl.col("mirrored").cast(pl.Boolean),
            pl.col("gameId").cast(pl.Int32),
            pl.col("playId").cast(pl.Int32),
            pl.col("frameId").cast(pl.Int32),
        ]
    )
    return df


def main():
    for split in ["test", "val", "train"]:
        feature_df = _read_features(split)
        tgt_df = _read_targets(split)

        for model_type in ["zoo", "transformer"]:
            print(f"Creating dataset for {model_type=}, {split=}...")
            tic = time.time()

            dataset = BDB2024_Dataset(model_type, feature_df, tgt_df)

            out_dir = DATASET_DIR / model_type
            out_dir.mkdir(exist_ok=True, parents=True)

            with open(out_dir / f"{split}_dataset.pkl", "wb") as f:
                pickle.dump(dataset, f)

            print(f"Took {(time.time() - tic)/60:.1f} mins")

        # free between splits
        del feature_df, tgt_df
        gc.collect()
        _malloc_trim()


if __name__ == "__main__":
    # On Colab/Linux, fork is typically fastest for this workload.
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    main()
