"""
Generate Results Summary and Analysis

This module performs comprehensive analysis of trained model performance, generating
publication-ready figures, tables, and metrics for comparing different architectures.

Key Analyses:
1. Overall Performance Comparison: Calculates Mean Absolute Error (MAE) for yards
   gained prediction across all data splits for each model architecture.

2. Event-Type Breakdown: Analyzes model performance at different game moments
   (snap, handoff, tackle, etc.) to understand where models excel or struggle.

3. Temporal Analysis: Examines how prediction accuracy changes as plays progress,
   measuring performance at different frames before the tackle event.

4. Model Scaling Analysis: Compares all trained models across different configurations
   to understand how each architecture responds to increased model capacity.

5. Computational Efficiency: Calculates FLOPs for inference to compare computational
   costs across different model sizes and architectures.

Outputs:
- results/results.csv: Comprehensive metrics table for all analyses
- results/model_comparison.json: Details of all trained model configurations
- results/frame_difference_plot.png: Temporal performance visualization
- results/model_scaling_plot.png: Model capacity vs. performance comparison

Usage:
    uv run python src/generate_results_summary.py
    uv run python src/generate_results_summary.py --models-dir models_norm --prepped-data-dir data/split_prepped_data_extra --num-features 7
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from calflops import calculate_flops

from models import LitModel


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Global config - set by parse_args()
MODELS_BASE_DIR = None
PREPPED_DATA_DIR = None
NUM_FEATURES = 7  # Number of features for transformer (will be set by args)


def generate_results_if_missing(checkpoint_path: Path, results_path: Path, model_type: str):
    """Generate predictions for a model if results file doesn't exist."""
    if results_path.exists():
        return

    print(f"Results file missing for {model_type}, generating predictions...")

    from train import predict_model_as_df

    # Generate predictions
    preds_df = predict_model_as_df(ckpt_path=checkpoint_path, devices=[0])

    # Save results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(results_path, compression="zstd", compression_level=22)
    print(f"Saved results to {results_path}")


def load_results() -> pl.DataFrame:
    """Load and combine results from all models, including per-frame events from tracking data."""
    print("Loading model results...")

    # Find all .results.parquet files in the models directory
    results_files = list(MODELS_BASE_DIR.glob("**/*.results.parquet"))

    if not results_files:
        raise FileNotFoundError(f"No .results.parquet files found in {MODELS_BASE_DIR}. Please train models first.")

    print(f"  Found {len(results_files)} results files")

    # Load and combine all results
    results_dfs = []
    for results_file in results_files:
        try:
            df = pl.read_parquet(results_file)
            results_dfs.append(df)
            print(f"  Loaded: {results_file.relative_to(MODELS_BASE_DIR)}")
        except Exception as e:
            print(f"  Warning: Failed to load {results_file}: {e}")

    if not results_dfs:
        raise FileNotFoundError("Failed to load any results files.")

    results_df = pl.concat(results_dfs, how="diagonal")

    # Load tracking data to get per-frame events
    print("Loading tracking data for per-frame events...")
    tracking_df = pl.read_parquet(f"{PREPPED_DATA_DIR}/*_features.parquet")

    # Join with tracking data to get per-frame events
    # Use ball carrier position when available, otherwise use first offensive player (e.g., QB)
    ball_carrier_tracking = tracking_df.filter(pl.col("is_ball_carrier") == 1)

    # For plays without ball carrier, use QB or first offensive player
    fallback_tracking = (
        tracking_df.filter(pl.col("side") == 1)
        .group_by(["gameId", "playId", "frameId", "mirrored"])
        .agg([
            pl.col("x").filter(pl.col("position") == "QB").first().alias("x_qb"),
            pl.col("y").filter(pl.col("position") == "QB").first().alias("y_qb"),
            pl.col("x").first().alias("x_first"),
            pl.col("y").first().alias("y_first"),
            pl.col("event").first().alias("event"),
        ])
        .with_columns([
            pl.coalesce(["x_qb", "x_first"]).alias("x"),
            pl.coalesce(["y_qb", "y_first"]).alias("y"),
        ])
        .select(["x", "y", "gameId", "playId", "frameId", "mirrored", "event"])
    )

    # Combine ball carrier and fallback tracking
    player_tracking = pl.concat([
        ball_carrier_tracking.select(["x", "y", "gameId", "playId", "frameId", "mirrored", "event"]),
        fallback_tracking.join(
            ball_carrier_tracking.select(["gameId", "playId", "frameId", "mirrored"]),
            on=["gameId", "playId", "frameId", "mirrored"],
            how="anti",
        ),
    ]).rename({"x": "ball_carrier_x", "y": "ball_carrier_y"})

    results_df = results_df.join(
        player_tracking,
        on=["gameId", "playId", "frameId", "mirrored"],
        how="left",
    )

    # Filter to mirrored=False to avoid double-counting predictions
    #
    # During training, we augment data by horizontally flipping each play (data augmentation).
    # This gives us 2× more training examples from the same data, helping the model generalize.
    #
    # Example: Original play has ball carrier running right → tackle at x=30, y=25
    #          Mirrored play has ball carrier running left → tackle at x=70, y=25 (x is flipped)
    #
    # When evaluating, we only count each unique play once to avoid inflating our metrics.
    # Both the original and mirrored versions produce predictions, but we only evaluate
    # the original (mirrored=False) to get true performance on unique plays.
    results_df = results_df.filter(pl.col("mirrored") == False)
    print(f"  Loaded {len(results_df):,} predictions (mirrored=False only)")
    return results_df


def _calculate_mae_for_df(df: pl.DataFrame) -> float | None:
    """
    Helper to calculate MAE (Mean Absolute Error) for yards gained prediction.

    Args:
        df: DataFrame with columns: yards_gained, expected_yards

    Returns:
        MAE in yards, rounded to 2 decimal places, or None if df is empty
    """
    if len(df) == 0:
        return None

    mae = df.select(
        (pl.col("yards_gained") - pl.col("expected_yards")).abs().mean()
    ).item()
    return round(mae, 2) if mae is not None else None


def _calculate_improvement_metrics(zoo_ade: float, transformer_ade: float) -> tuple[float, float]:
    """
    Calculate improvement metrics comparing Zoo baseline to Transformer model.

    Args:
        zoo_ade: ADE for Zoo model (baseline)
        transformer_ade: ADE for Transformer model

    Returns:
        Tuple of (improvement_pct, improvement_yards)
        - improvement_pct: Percentage improvement (positive = better)
        - improvement_yards: Absolute yards improvement
    """
    improvement_pct = round((zoo_ade - transformer_ade) / zoo_ade * 100, 1)
    improvement_yards = round(zoo_ade - transformer_ade, 2)
    return improvement_pct, improvement_yards


def calculate_results(results_df: pl.DataFrame) -> list[dict]:
    """Calculate results for all splits and events."""
    print("\nCalculating results...")

    results = []

    # Main splits
    for split in ["train", "val", "test"]:
        split_df = results_df.filter(pl.col("dataset_split") == split)

        row = {"split": split, "metric": "mae_yards"}

        # Only calculate for transformer (no zoo model in 23-entity-football branch)
        model_df = split_df.filter(pl.col("model_type") == "transformer")
        row["transformer"] = _calculate_mae_for_df(model_df)

        row["n_plays"] = split_df.select(pl.struct(["gameId", "playId"]).n_unique()).item()
        row["n_frames"] = split_df.select(pl.len()).item()

        results.append(row)

    # Test event breakdowns (using per-frame events)
    print("Calculating test set event breakdowns (per-frame events)...")
    test_df = results_df.filter(pl.col("dataset_split") == "test")
    events = sorted(test_df.filter(pl.col("event").is_not_null())["event"].unique().to_list())

    event_results = []
    for event in events:
        # Use per-frame events (all frames with this event)
        event_df = test_df.filter(pl.col("event") == event)

        # Skip events with too few plays
        n_plays = event_df.select(pl.struct(["gameId", "playId"]).n_unique()).item()
        if n_plays < 100:
            continue

        row = {"split": f"test-event-{event}", "metric": "mae_yards"}

        # Only calculate for transformer (no zoo model in 23-entity-football branch)
        model_df = event_df.filter(pl.col("model_type") == "transformer")
        row["transformer"] = _calculate_mae_for_df(model_df)

        row["n_plays"] = n_plays
        row["n_frames"] = event_df.select(pl.len()).item()
        row["_avg_frameId"] = round(event_df["frameId"].mean(), 1)  # For sorting only

        event_results.append(row)

    # Sort event results by avg_frameId, then remove the sorting key
    event_results.sort(key=lambda x: x["_avg_frameId"])
    for row in event_results:
        del row["_avg_frameId"]
    results.extend(event_results)

    return results


def calculate_frame_difference_results(results_df: pl.DataFrame) -> tuple[list[dict], pl.DataFrame]:
    """
    Calculate results by frame difference from tackle (test set only).

    Returns:
        tuple: (list of result dicts, DataFrame for plotting)
    """
    print("\nCalculating frame-difference breakdown (test set only)...")

    # Check if tackle_frameId exists - skip if not available
    if "tackle_frameId" not in results_df.columns:
        print("  Skipping: tackle_frameId column not available in data")
        return [], pl.DataFrame()

    frame_diff_df = (
        results_df.with_columns(
            frame_difference_from_tackle=(pl.col("tackle_frameId") - pl.col("frameId")),
        )
        .with_columns(
            frame_difference_from_tackle_cat=(
                pl.col("frame_difference_from_tackle").cut(
                    breaks=range(0, 31, 5),
                    labels=["after tackle", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"],
                    left_closed=True,
                )
            )
        )
        .filter(pl.col("dataset_split") == "test")
        .group_by(["model_type", "frame_difference_from_tackle_cat"])
        .agg(
            order=pl.col("frame_difference_from_tackle").mean() * -1,
            n_frames=pl.len(),
            n_plays=pl.struct(["gameId", "playId"]).n_unique(),
            mae_yards=(pl.col("yards_gained") - pl.col("expected_yards")).abs().mean().round(2),
        )
        .sort("frame_difference_from_tackle_cat")
    )

    # Convert to results format for JSON
    results = []
    categories = sorted(frame_diff_df["frame_difference_from_tackle_cat"].unique().to_list())

    for category in categories:
        cat_df = frame_diff_df.filter(pl.col("frame_difference_from_tackle_cat") == category)

        row = {"split": f"test-frames-before-tackle-{category}", "metric": "mae_yards"}

        # Only calculate for transformer (no zoo model in 23-entity-football branch)
        model_data = cat_df.filter(pl.col("model_type") == "transformer")
        if len(model_data) > 0:
            row["transformer"] = model_data["mae_yards"].item()

        # Get n_plays and n_frames
        first_row = cat_df.row(0, named=True)
        row["n_plays"] = int(first_row["n_plays"])
        row["n_frames"] = int(first_row["n_frames"])

        results.append(row)

    return results, frame_diff_df


def generate_frame_difference_plot(frame_diff_df: pl.DataFrame) -> None:
    """Generate and save frame-difference plot."""
    print("\nGenerating frame-difference plot...")

    plot_path = RESULTS_DIR / "frame_difference_plot.png"

    # Create placeholder plot if no data
    if len(frame_diff_df) == 0:
        print("  Creating placeholder: no frame-difference data available")
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "No frame-difference data available\n(tackle_frameId column not present)",
                 ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.title("Model Performance by Frames Before Tackle", fontsize=16)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved placeholder: {plot_path}")
        return

    # Convert to pandas for plotting
    frame_diff_df_pandas = frame_diff_df.to_pandas()

    # Create the line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=frame_diff_df_pandas,
        x="frame_difference_from_tackle_cat",
        y="mae_yards",
        hue="model_type",
        marker="o",
    )

    # Flip the x-axis
    plt.gca().invert_xaxis()

    # Customize the plot
    plt.title("Model Performance by Frames Before Tackle", fontsize=16)
    plt.xlabel("Frames Before Tackle", fontsize=12)
    plt.ylabel("Mean Absolute Error (yards)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model Type", title_fontsize="12", fontsize="10")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {plot_path}")


def find_all_model_checkpoints(require_results: bool = True) -> list[dict]:
    """
    Find all model checkpoints and group by configuration.

    Args:
        require_results: If True, only return checkpoints that have results files.
                        If False, return all checkpoints.

    Returns:
        list[dict]: List of config dicts with model_type, model_dim, num_layers, and best checkpoint path.
    """
    models_base = MODELS_BASE_DIR
    configs = []

    # Only check for transformer models (no zoo model in 23-entity-football branch)
    for model_type in ["transformer"]:
        model_dir = models_base / model_type
        if not model_dir.exists():
            continue

        # Find all config directories (e.g., M128_L2_LR1e-04)
        for config_dir in model_dir.iterdir():
            if not config_dir.is_dir() or not config_dir.name.startswith("M"):
                continue

            # Parse config from directory name
            match = re.match(r"M(\d+)_L(\d+)_LR", config_dir.name)
            if not match:
                continue

            model_dim = int(match.group(1))
            num_layers = int(match.group(2))

            # Find best checkpoint (lowest val_loss)
            checkpoints_dir = config_dir / "checkpoints"
            if not checkpoints_dir.exists():
                continue

            checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
            if not checkpoint_files:
                continue

            # Parse val_loss from filename and find best
            best_checkpoint = None
            best_val_loss = float("inf")

            for ckpt_file in checkpoint_files:
                # Parse: epoch=X-val_loss=Y.YYY.ckpt
                val_loss_match = re.search(r"val_loss=([\d.]+?)\.ckpt", ckpt_file.name)
                if val_loss_match:
                    val_loss = float(val_loss_match.group(1))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = ckpt_file

            if best_checkpoint:
                # Find corresponding results file
                results_file = best_checkpoint.with_suffix(".results.parquet")

                # Add config if results exist or if we don't require them
                if not require_results or results_file.exists():
                    configs.append(
                        {
                            "model_type": model_type,
                            "model_dim": model_dim,
                            "num_layers": num_layers,
                            "checkpoint_path": str(best_checkpoint),
                            "results_path": str(results_file),
                            "val_loss": best_val_loss,
                        }
                    )

    return configs


def compute_model_metrics(checkpoint_path: str, model_type: str) -> dict:
    """
    Compute params and FLOPs for a single model checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint file.
        model_type (str): 'zoo' or 'transformer'.

    Returns:
        dict: Metrics including params and inference_flops.
    """
    # Load model
    lit_model = LitModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model = lit_model.model
    model.eval()

    # Create dummy input shape
    if model_type == "transformer":
        input_shape = (1, 22, NUM_FEATURES)
    else:  # zoo
        input_shape = (1, 10, 11, 10)

    # Calculate params
    params = int(lit_model.hparams["params"])

    # Calculate FLOPs using calflops
    # Note: We use calflops instead of fvcore because it properly counts
    # transformer attention operations (scaled_dot_product_attention),
    # which are critical for accurate FLOP comparison between models.
    try:
        flops, macs, _ = calculate_flops(
            model=model,
            input_shape=input_shape,
            print_results=False,
            output_as_string=False,
        )
        inference_flops = int(flops)
    except Exception:
        inference_flops = None

    return {"params": params, "inference_flops": inference_flops}


def compute_test_mae(results_path: str) -> float:
    """
    Compute test set MAE from results parquet file.

    Args:
        results_path (str): Path to results parquet file.

    Returns:
        float: Test set MAE in yards.
    """
    df = pl.read_parquet(results_path)
    test_df = df.filter((pl.col("dataset_split") == "test") & (pl.col("mirrored") == False))

    mae = test_df.select(
        (pl.col("yards_gained") - pl.col("expected_yards")).abs().mean()
    ).item()

    return float(mae)


def compute_model_comparison() -> list[dict]:
    """
    Compute comprehensive comparison of all trained models.

    Returns:
        list[dict]: List of model configs with params, FLOPs, and test ADE.
    """
    print("\nComputing comprehensive model comparison...")

    # Find all checkpoints
    configs = find_all_model_checkpoints()
    print(f"  Found {len(configs)} model configurations")

    results = []

    for i, config in enumerate(configs, 1):
        print(
            f"  [{i}/{len(configs)}] Processing {config['model_type']} "
            f"M{config['model_dim']}_L{config['num_layers']}..."
        )

        # Compute metrics
        metrics = compute_model_metrics(config["checkpoint_path"], config["model_type"])
        test_mae = compute_test_mae(config["results_path"])

        results.append(
            {
                "model_type": config["model_type"],
                "model_dim": config["model_dim"],
                "num_layers": config["num_layers"],
                "params": metrics["params"],
                "inference_flops": metrics["inference_flops"],
                "test_mae_yards": round(test_mae, 2),
                "val_loss": round(config["val_loss"], 3),
            }
        )

    # Sort by model_type, then params
    results.sort(key=lambda x: (x["model_type"], x["params"]))

    return results


def generate_model_scaling_plot(model_comparison: list[dict]) -> None:
    """
    Generate model scaling plot showing Test ADE vs FLOPs.

    This visualization supports the "Model Selection and Architectural Scaling"
    section of the paper, demonstrating how Zoo and Transformer architectures
    scale with computational budget.
    """
    print("\nGenerating model scaling plot...")

    # Convert to DataFrame for easier manipulation
    df = pl.DataFrame(model_comparison).to_pandas()

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Define colors and markers (only transformer in 23-entity-football branch)
    colors = {"transformer": "#1F77B4"}
    markers = {"transformer": "o"}

    # Plot: Test ADE vs FLOPs (only transformer models)
    for model_type in ["transformer"]:
        data = df[df["model_type"] == model_type].sort_values("inference_flops")
        ax.plot(
            data["inference_flops"],
            data["test_mae_yards"],
            marker=markers[model_type],
            markersize=8,
            linewidth=2,
            label=model_type.capitalize(),
            color=colors[model_type],
            alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference FLOPs (log scale)", fontsize=12)
    ax.set_ylabel("Test MAE (yards) - Lower is Better", fontsize=12)
    ax.set_title("Model Scaling: Test MAE vs FLOPs", fontsize=14, fontweight="bold")
    ax.legend(title="Architecture", fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Adjust layout and save
    plt.tight_layout()
    plot_path = RESULTS_DIR / "model_scaling_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {plot_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate results summary and analysis")
    parser.add_argument("--models-dir", type=str, default="models_norm", help="Models directory (default: models_norm)")
    parser.add_argument("--prepped-data-dir", type=str, default="data/split_prepped_data_extra",
                       help="Prepared data directory (default: data/split_prepped_data_extra)")
    parser.add_argument("--num-features", type=int, default=7,
                       help="Number of input features for transformer (default: 7)")
    parser.add_argument("--best-only", action="store_true",
                       help="Only generate results for the checkpoint with lowest validation loss")
    return parser.parse_args()


def main():
    """Generate results summary."""
    global MODELS_BASE_DIR, PREPPED_DATA_DIR, NUM_FEATURES

    args = parse_args()

    # Set global config from args
    NUM_FEATURES = args.num_features
    PREPPED_DATA_DIR = Path(args.prepped_data_dir)

    # Determine models base directory (check Google Drive first)
    gdrive_models_path = Path(f"/content/drive/MyDrive/SportsTrackingTransformer/{args.models_dir}")
    local_models_path = Path(args.models_dir)

    if gdrive_models_path.exists():
        MODELS_BASE_DIR = gdrive_models_path
        print(f"Using Google Drive for models: {MODELS_BASE_DIR}")
    else:
        MODELS_BASE_DIR = local_models_path
        print(f"Using local path for models: {MODELS_BASE_DIR}")

    print(f"Using prepped data from: {PREPPED_DATA_DIR}")
    print(f"Transformer input features: {NUM_FEATURES}")

    print("\n" + "=" * 60)
    print("GENERATING RESULTS")
    print("=" * 60)

    # If best-only flag is set, find and process only the best checkpoint
    if args.best_only:
        print("\nFinding best checkpoint (lowest validation loss)...")
        configs = find_all_model_checkpoints(require_results=False)
        if not configs:
            print("No model checkpoints found!")
            return

        # Find the config with lowest validation loss
        best_config = min(configs, key=lambda x: x["val_loss"])
        print(f"Best model: {best_config['model_type']} "
              f"M{best_config['model_dim']}_L{best_config['num_layers']}")
        print(f"Validation loss: {best_config['val_loss']:.3f}")
        print(f"Checkpoint: {best_config['checkpoint_path']}")

        # Generate results only for best checkpoint
        results_file = Path(best_config['results_path'])
        if results_file.exists():
            print(f"\n✓ Results already exist: {results_file}")
            return

        print(f"\nGenerating results for best checkpoint...")
        from train import predict_model_as_df

        # Generate predictions
        preds_df = predict_model_as_df(ckpt_path=Path(best_config['checkpoint_path']), devices=[0])

        # Save results
        results_file.parent.mkdir(parents=True, exist_ok=True)
        preds_df.write_parquet(results_file, compression="zstd", compression_level=22)
        print(f"✓ Results saved to: {results_file}")
        return

    results_df = load_results()
    results = calculate_results(results_df)

    # Add frame-difference results (test only)
    frame_diff_results, frame_diff_df = calculate_frame_difference_results(results_df)
    results.extend(frame_diff_results)

    # Generate frame-difference plot
    generate_frame_difference_plot(frame_diff_df)

    # Save results CSV
    results_csv_path = RESULTS_DIR / "results.csv"
    results_pl_df = pl.DataFrame(results)
    results_pl_df.write_csv(results_csv_path)
    print(f"\n✓ Saved: {results_csv_path}")

    # Compute and save comprehensive model comparison
    model_comparison = compute_model_comparison()
    comparison_path = RESULTS_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(model_comparison, f, indent=2)
    print(f"\n✓ Saved: {comparison_path} ({len(model_comparison)} models)")

    # Generate model scaling plot
    generate_model_scaling_plot(model_comparison)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    # Print comprehensive model performance table
    print("\n" + "=" * 60)
    print("ALL MODELS PERFORMANCE (MAE in yards)")
    print("=" * 60)

    # Create a structured table of all models' performance
    print(f"\n{'Model':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Val Loss':<10}")
    print("-" * 65)

    # Sort by test MAE (best first)
    sorted_models = sorted(model_comparison, key=lambda x: x['test_mae_yards'])

    for i, model in enumerate(sorted_models):
        model_name = f"{model['model_type']}_M{model['model_dim']}_L{model['num_layers']}"

        # Get train/val MAE from results
        train_mae = val_mae = None
        for row in results:
            if row['split'] == 'train' and model['model_type'] in row:
                train_mae = row[model['model_type']]
            elif row['split'] == 'val' and model['model_type'] in row:
                val_mae = row[model['model_type']]

        train_str = f"{train_mae:.2f}" if train_mae is not None else "N/A"
        val_str = f"{val_mae:.2f}" if val_mae is not None else "N/A"
        test_str = f"{model['test_mae_yards']:.2f}"
        val_loss_str = f"{model['val_loss']:.3f}"

        # Mark the best model on test set
        prefix = "→ " if i == 0 else "  "
        print(f"{prefix}{model_name:<23} {train_str:<10} {val_str:<10} {test_str:<10} {val_loss_str:<10}")

    # Explicitly state the best model
    best_model = sorted_models[0]
    print("\n" + "=" * 60)
    print(f"BEST MODEL ON TEST SET: {best_model['model_type']}_M{best_model['model_dim']}_L{best_model['num_layers']}")
    print(f"Test MAE: {best_model['test_mae_yards']:.2f} yards")
    print(f"Validation Loss: {best_model['val_loss']:.3f}")
    print("=" * 60)

    # Print test set summary
    test_row = next(r for r in results if r["split"] == "test")
    print(f"\nTest Set Overall:")
    if "transformer" in test_row:
        print(f"  Transformer: {test_row['transformer']:.2f} yards")

    print(f"\nTest Set Events:")
    for row in results:
        if row["split"].startswith("test-event-"):
            event_name = row["split"].replace("test-event-", "")
            if "transformer" in row:
                print(f"  {event_name:20s}: {row['transformer']:5.2f} yards")

    print(f"\nTest Set Frame Differences:")
    for row in results:
        if row["split"].startswith("test-frames-before-tackle-"):
            frame_cat = row["split"].replace("test-frames-before-tackle-", "")
            if "transformer" in row:
                print(f"  {frame_cat:15s}: {row['transformer']:5.2f} yards")


if __name__ == "__main__":
    main()
