"""
Training Script for NFL Big Data Bowl 2024 Yards Gained Prediction Models

This module handles the training process for yards gained prediction models. It includes
functions for loading datasets, predicting using trained models, and conducting
hyperparameter searches.

Functions:
    predict_model_as_df: Generate predictions using a trained model and return results as a DataFrame
    train_model: Train a single model with specified hyperparameters
    main: Main execution function for hyperparameter search and model training

Classes:
    None (uses classes from other modules)
"""

import os
import random
import re
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import lightning.pytorch.callbacks as callbacks
import numpy as np
import polars as pl
import psutil
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import BDB2024_Dataset, load_datasets, MIN_YARDS, NUM_YARDS_CLASSES, PREPPED_DATA_DIR
from models import LitModel

# Use Google Drive for checkpoints if available (Colab), otherwise local
GDRIVE_MODELS_PATH = Path("/content/drive/MyDrive/SportsTrackingTransformer/models_norm_football")
LOCAL_MODELS_PATH = Path("models_norm_football")

if GDRIVE_MODELS_PATH.parent.exists():
    MODELS_PATH = GDRIVE_MODELS_PATH
    print(f"Using Google Drive for checkpoints: {MODELS_PATH}")
else:
    MODELS_PATH = LOCAL_MODELS_PATH
    print(f"Using local path for checkpoints: {MODELS_PATH}")

MODELS_PATH.mkdir(exist_ok=True, parents=True)


def get_optimal_dataloader_config():
    """
    Determine optimal num_workers and batch_size based on available system resources.

    Detects GPU type and system RAM to configure DataLoader settings:
    - A100 instances (157GB RAM): 10-12 workers, larger batches
    - L4 instances (53GB RAM): 4-6 workers, smaller batches
    - Other instances: Conservative defaults

    Returns:
        dict: Configuration with 'num_workers_train', 'num_workers_pred', 'batch_size_multiplier'
    """
    # Get total system RAM in GB
    total_ram_gb = psutil.virtual_memory().total / (1024**3)

    # Get GPU info if available
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except:
            pass

    print(f"Detected system RAM: {total_ram_gb:.1f} GB")
    if gpu_name:
        print(f"Detected GPU: {gpu_name}")

    # Configure based on RAM (more reliable than GPU detection)
    if total_ram_gb >= 140:  # A100 instance or similar (157GB)
        num_workers_train = 12
        num_workers_pred = 10
        batch_size_multiplier = 1.0
        print("Using A100-optimized settings: 12/10 workers, standard batch sizes")
    elif total_ram_gb >= 48:  # L4 instance (53GB) or similar
        num_workers_train = 4
        num_workers_pred = 2
        batch_size_multiplier = 0.5
        print("Using L4-optimized settings: 4/2 workers, 50% batch sizes")
    elif total_ram_gb >= 24:  # Mid-tier instance
        num_workers_train = 3
        num_workers_pred = 2
        batch_size_multiplier = 0.4
        print("Using mid-tier settings: 3/2 workers, 40% batch sizes")
    else:  # Low RAM instance (< 24GB)
        num_workers_train = 2
        num_workers_pred = 1
        batch_size_multiplier = 0.3
        print("Using low-memory settings: 2/1 workers, 30% batch sizes")

    return {
        'num_workers_train': num_workers_train,
        'num_workers_pred': num_workers_pred,
        'batch_size_multiplier': batch_size_multiplier,
    }


# Get optimal configuration based on system resources
DATALOADER_CONFIG = get_optimal_dataloader_config()

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict_model_as_df(model: LitModel = None, ckpt_path: Path = None, devices=1) -> pl.DataFrame:
    """
    Generate predictions using a trained model and return results as a DataFrame.

    Args:
        model (LitModel, optional): Trained model instance. Defaults to None.
        ckpt_path (Path, optional): Path to model checkpoint. Defaults to None.
        devices (int or list): Devices to use for prediction. Defaults to 1.

    Returns:
        pl.DataFrame: DataFrame containing model predictions and metadata.

    Raises:
        AssertionError: If neither model nor ckpt_path is provided, or if multiple devices are specified.
    """
    assert model is not None or ckpt_path is not None, "Must provide either model or ckpt_path"
    if isinstance(devices, list):
        assert len(devices) == 1, "Only one device should be used for prediction"

    if model is None:
        model = LitModel.load_from_checkpoint(ckpt_path)

    # Load datasets
    train_ds: BDB2024_Dataset = load_datasets(model.model_type, split="train")
    val_ds: BDB2024_Dataset = load_datasets(model.model_type, split="val")
    test_ds: BDB2024_Dataset = load_datasets(model.model_type, split="test")

    # Create unshuffled dataloaders for prediction
    # Use dynamic num_workers based on system resources
    num_workers_pred = DATALOADER_CONFIG['num_workers_pred']
    dataloaders = {
        "train": DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=num_workers_pred),
        "val": DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=num_workers_pred),
        "test": DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=num_workers_pred),
    }

    # Yard values for computing expected yards from probability distribution
    yard_values = np.arange(MIN_YARDS, MIN_YARDS + NUM_YARDS_CLASSES, dtype=np.float32)

    pred_dfs = []
    for split, dataloader in dataloaders.items():
        # Generate predictions (logits)
        pred_trainer = Trainer(devices=devices, logger=False, enable_model_summary=False)
        preds = pred_trainer.predict(model, dataloaders=dataloader, ckpt_path=ckpt_path)
        logits: np.ndarray = torch.concat(preds, dim=0).cpu().numpy()

        # Convert logits to probabilities and compute expected yards
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))  # softmax with numerical stability
        probs = probs / probs.sum(axis=1, keepdims=True)
        expected_yards = (probs * yard_values).sum(axis=1)
        predicted_class = logits.argmax(axis=1)

        # Prepare metadata - load target data from parquet since tgt_df_partition is cleared
        tgt_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_targets.parquet")

        dataset: BDB2024_Dataset = dataloader.dataset
        ds_keys = np.array(dataset.keys)

        assert logits.shape[0] == ds_keys.shape[0], f"Pred Shape: {logits.shape}, Keys Shape: {ds_keys.shape}"

        # Create prediction DataFrame
        pred_df = (
            tgt_df.join(
                pl.DataFrame(
                    {
                        "gameId": ds_keys[:, 0],
                        "playId": ds_keys[:, 1],
                        "mirrored": ds_keys[:, 2],
                        "frameId": ds_keys[:, 3],
                        "dataset_split": split,
                        "expected_yards": expected_yards.round(2),
                        "predicted_class": predicted_class,
                    },
                    schema_overrides={"mirrored": bool},
                ),
                on=["gameId", "playId", "mirrored", "frameId"],
                how="inner",
            )
            # add model hparams to pred df
            .with_columns(**{k: pl.lit(v) for k, v in model.hparams.items()})
        )

        assert pred_df.shape[0] == len(dataset)
        pred_dfs.append(pred_df)

    return pl.concat(pred_dfs, how="vertical")


def get_epoch_val_loss_from_ckpt(ckpt_path: Path) -> tuple[int, float]:
    """
    Extract epoch number and validation loss from checkpoint filename.

    Parses checkpoint filenames that follow the pattern 'epoch={N}-val_loss={X.XXX}.ckpt'
    to extract training metadata for resuming or comparison.

    Args:
        ckpt_path (Path): Path to the checkpoint file.

    Returns:
        tuple[int, float]: A tuple containing:
            - epoch (int): The epoch number when checkpoint was saved (-1 if not found)
            - val_loss (float): The validation loss value (inf if not found)

    Example:
        >>> get_epoch_val_loss_from_ckpt(Path("epoch=10-val_loss=2.543.ckpt"))
        (10, 2.543)
    """
    ckpt_path = Path(ckpt_path)

    val_loss_pattern = re.compile(r"epoch=(\d+)-val_loss=([\d\.]+)")
    match = val_loss_pattern.search(ckpt_path.name)
    if match:
        try:
            epoch = int(match.group(1))
            val_loss = float(match.group(2).rstrip("."))
            return epoch, val_loss
        except ValueError:
            print(f"Warning: Invalid epoch or val_loss in checkpoint name: {ckpt_path.name}")
            return -1, float("inf")
    return -1, float("inf")


def train_model(
    model_type,
    batch_size,
    model_dim,
    num_layers,
    learning_rate,
    dropout,
    device=0,
    dbg_run=False,
    skip_existing=False,
    skip_if_trained=False,
    min_epochs=40,
    patience=5,
):
    """
    Train a single model with specified hyperparameters.

    This function handles the complete training pipeline including:
    - Checkpoint resumption from previous runs
    - Model initialization with hyperparameters
    - Dataset loading and dataloader creation
    - Training with early stopping and model checkpointing
    - Prediction generation and saving for the best model

    Args:
        model_type (str): Type of model to train ('transformer' or 'zoo').
        batch_size (int): Batch size for training (typically 256).
        model_dim (int): Dimension of the model's internal representations (32, 128, or 512).
        num_layers (int): Number of layers in the model (1, 2, 4, or 8).
        learning_rate (float): Learning rate for AdamW optimizer (typically 1e-4).
        dropout (float): Dropout rate for regularization (typically 0.3).
        device (int, optional): GPU device index to use (-1 for CPU). Defaults to 0.
        dbg_run (bool, optional): Whether to run in debug mode with profiling. Defaults to False.
        skip_existing (bool, optional): Skip training if checkpoint exists. Defaults to False.
        skip_if_trained (bool, optional): Skip training if checkpoint has min_epochs or more. Defaults to False.
        min_epochs (int, optional): Minimum epochs required for skip_if_trained. Defaults to 40.
        patience (int, optional): Early stopping patience in epochs. Defaults to 5.

    Returns:
        LitModel: Trained model instance with best validation performance.

    Note:
        Training automatically resumes from the best checkpoint if one exists,
        unless skip_existing=True in which case training is skipped entirely.
    """
    # Set up logger and trainer for full run
    logger = TensorBoardLogger(
        save_dir=MODELS_PATH,
        name=model_type,
        log_graph=False,
        default_hp_metric=False,
        version=f"M{model_dim}_L{num_layers}_LR{learning_rate:.0e}",
    )

    # Check for existing checkpoint with best val_loss
    ckpt_dir = Path(logger.log_dir) / "checkpoints"
    existing_ckpt = None
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            # Find the checkpoint with the lowest val_loss
            best_ckpt = min(ckpts, key=lambda x: get_epoch_val_loss_from_ckpt(x)[1])
            existing_ckpt = str(best_ckpt)
            print(f"Resuming training from best checkpoint: {existing_ckpt}")

    # initialize model
    if existing_ckpt is not None:
        lit_model = LitModel.load_from_checkpoint(existing_ckpt)
        curr_epoch, _ = get_epoch_val_loss_from_ckpt(existing_ckpt)
    else:
        lit_model = LitModel(
            model_type,
            batch_size=batch_size,
            model_dim=model_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            dropout=dropout,
        )
        curr_epoch = 0

    # if skip_existing and checkpoint exists, skip re-training
    if skip_existing and existing_ckpt is not None:
        print(f"Skipping training as checkpoint exists: {existing_ckpt}")
        return lit_model

    # if skip_if_trained and checkpoint has reached min_epochs, skip re-training
    if skip_if_trained and existing_ckpt is not None:
        if curr_epoch >= min_epochs:
            print(f"Skipping training as checkpoint has reached {curr_epoch} epochs (>= {min_epochs}): {existing_ckpt}")
            return lit_model
        else:
            print(f"Checkpoint at {curr_epoch} epochs (< {min_epochs}), continuing training: {existing_ckpt}")

    # Load preprocessed datasets specific to model type
    # Zoo and Transformer models require different feature formats
    train_ds: BDB2024_Dataset = load_datasets(model_type, split="train")
    val_ds: BDB2024_Dataset = load_datasets(model_type, split="val")

    # Create dataloaders with optimized settings
    # Training: smaller batch size, shuffled for better generalization
    # Validation: larger batch size (1024), no shuffle for consistent evaluation
    # num_workers dynamically set based on system resources (A100: 12, L4: 6, etc.)
    num_workers_train = DATALOADER_CONFIG['num_workers_train']
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers_train)
    val_dataloader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True, num_workers=num_workers_train)

    # Set up devices
    devices = [device] if device >= 0 else [0, 1]  # if device is specified, use it, otherwise pick 1 gpu to use

    if dbg_run:
        # Debug run with limited epochs and profiling
        dbg_trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            profiler="simple",
            fast_dev_run=True,
            enable_model_summary=False,
            num_sanity_val_steps=3,
        )
        dbg_trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    trainer = Trainer(
        max_epochs=200,
        accelerator="gpu",
        profiler=None,
        logger=logger,
        devices=devices,
        sync_batchnorm=True,
        enable_model_summary=True,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=patience),
            callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, filename="{epoch}-{val_loss:.3f}"),
            callbacks.ModelSummary(max_depth=2),
        ],
    )

    # Train the model
    print(lit_model.get_hyperparams())
    logger.log_hyperparams(lit_model.get_hyperparams())
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=existing_ckpt)

    # Generate and save predictions for the best model
    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    preds_df = predict_model_as_df(lit_model, best_ckpt_path, devices[:1])
    preds_df.write_parquet(best_ckpt_path.with_suffix(".results.parquet"), compression="zstd", compression_level=22)

    return lit_model


def main(args):
    """
    Main execution function for hyperparameter search and model training.

    Args:
        args (Namespace): Command-line arguments.

    This function sets up the hyperparameter search space, shuffles the combinations,
    and trains models for each combination. It supports both exhaustive grid search
    and random search based on the provided arguments.
    """
    # Hyperparameter search space:
    # - lrs: Learning rate (1e-4 based on prior experimentation)
    # - model_dims: Model width (32, 64, 128) - # size of internal vector representation for each player in each layer
    # - num_layers: Model depth (1, 2, 4, 8) - number of stacked layers
    # - Custom additions: M64_L4, M64_L8 (added to test medium-width model)
    # - Excluded: M32_L8 (too deep for narrow model), M512 (removed entirely)
    #
    # Total: 11 configurations (2 model_dims × 4 num_layers + 2 custom - 1 excluded = 11)

    lrs = [1e-4]
    model_dims = [32, 128]
    num_layers = [1, 2, 4, 8]

    # Create gridsearch iterable
    gridsearch = list(product(model_dims, num_layers, lrs))
    # Filter out M32_L8 configuration
    gridsearch = [(M, L, LR) for M, L, LR in gridsearch if not (M == 32 and L == 8)]
    # Add M64 custom configurations after M32 models
    # Insert M64_L1, M64_L2, M64_L4, M64_L8 after M32 models (before M128 models)
    m32_count = sum(1 for M, L, LR in gridsearch if M == 32)
    gridsearch.insert(m32_count, (64, 1, 1e-4))
    gridsearch.insert(m32_count + 1, (64, 2, 1e-4))
    gridsearch.insert(m32_count + 2, (64, 4, 1e-4))
    gridsearch.insert(m32_count + 3, (64, 8, 1e-4))

    # Filter out L1 models if requested
    if args.skip_l1:
        gridsearch = [(M, L, LR) for M, L, LR in gridsearch if L != 1]
        print(f"Skipping L1 models - {len(gridsearch)} configurations remaining")

    if args.shuffle:
        random.shuffle(gridsearch)
    if args.reverse:
        gridsearch.reverse()

    if args.hparam_search_iters > 0:
        # Perform random search if hparam_search_iters is specified
        gridsearch = gridsearch[: args.hparam_search_iters]

    # Train models for each hyperparameter combination
    for M, L, LR in tqdm(gridsearch, desc="Hyperparam Gridsearch"):
        # Dynamic batch size based on model size to avoid OOM
        # Base sizes: 512 for M > 128, 1024 otherwise
        # Then apply system-based multiplier (A100: 1.0, L4: 0.75, etc.)
        base_batch_size = 512 if M > 128 else 1024
        batch_size = int(base_batch_size * DATALOADER_CONFIG['batch_size_multiplier'])
        # Ensure batch size is at least 64
        batch_size = max(64, batch_size)

        # Dynamic patience based on model size
        # Larger models need more patience to converge
        if M >= 128 and L >= 4:
            patience = max(args.patience, 10)  # At least 10 for large models
        elif M >= 128 or L >= 4:
            patience = max(args.patience, 8)   # At least 8 for medium-large models
        else:
            patience = args.patience            # Use default for smaller models

        # Dynamic dropout based on model size to prevent overfitting
        # Larger models (more params) need more regularization
        if M >= 128 and L >= 4:
            dropout = 0.35  # Medium-high dropout for large models (M128_L4, M128_L8)
        elif M >= 64 and L >= 4:
            dropout = 0.32  # Slightly higher dropout for M64_L4, M64_L8
        else:
            dropout = 0.3  # Standard dropout for smaller models

        train_model(
            model_type=args.model_type,
            batch_size=batch_size,
            model_dim=M,
            num_layers=L,
            learning_rate=LR,
            dropout=dropout,
            device=args.device,
            skip_existing=args.skip_existing,
            skip_if_trained=args.skip_if_trained,
            min_epochs=args.min_epochs,
            patience=patience,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=-1, help="GPU device to use (-1 for CPU)")
    parser.add_argument(
        "--hparam_search_iters", type=int, default=-1, help="Number of random hyperparameter combinations to try"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip training models that already have a checkpoint"
    )
    parser.add_argument(
        "--skip-if-trained", action="store_true", help="Skip training models with checkpoints that have >= min_epochs"
    )
    parser.add_argument(
        "--min-epochs", type=int, default=40, help="Minimum epochs for --skip-if-trained (default: 40)"
    )
    parser.add_argument("--shuffle", "-S", action="store_true", help="Shuffle the hyperparameter gridsearch")
    parser.add_argument("--reverse", "-R", action="store_true", help="Reverse the hyperparameter gridsearch")
    parser.add_argument("--skip-l1", action="store_true", help="Skip training all L1 (1-layer) models")
    parser.add_argument(
        "--model_type", type=str, default="transformer", help="Type of model to train ('transformer' or 'zoo')"
    )
    parser.add_argument("--patience", "-P", type=int, default=6, help="Early stopping patience")
    args = parser.parse_args()
    main(args)
