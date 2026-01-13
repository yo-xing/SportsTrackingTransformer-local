#!/usr/bin/env python3
"""
Compare 2-Week Models vs 18-Week Full Data Transformer

Compares performance of:
- 2-week Zoo model (best)
- 2-week Transformer model (best)
- 18-week Transformer model (best)

Generates comparison metrics and visualizations.

Usage:
    python compare_2week_vs_full.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from src.models import LitModel
from src.train import predict_model_as_df


# Model paths
MODELS_2WEEK_PATH = Path("models")  # 2-week models
MODELS_FULL_PATH = Path("models_full18weeks")  # 18-week models
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Data paths
DATA_2WEEK_PATH = Path("data/split_prepped_data_extra")
DATA_FULL_PATH = Path("data/split_prepped_data_extra_full18weeks")


def find_best_model(models_dir: Path, model_type: str) -> Path | None:
    """Find the best model checkpoint by lowest validation loss."""
    model_dir = models_dir / model_type
    if not model_dir.exists():
        return None

    best_ckpt = None
    best_val_loss = float('inf')

    # Search all version directories
    for version_dir in model_dir.iterdir():
        if not version_dir.is_dir():
            continue

        ckpt_dir = version_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        for ckpt_file in ckpt_dir.glob("*.ckpt"):
            # Extract val_loss from filename (e.g., epoch=10-val_loss=2.345.ckpt)
            match = re.search(r'val_loss=([\d.]+)', ckpt_file.name)
            if match:
                val_loss = float(match.group(1))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = ckpt_file

    return best_ckpt


def load_model_and_predict(ckpt_path: Path, data_path: Path, split: str = "test") -> pl.DataFrame:
    """Load model checkpoint and generate predictions."""
    print(f"Loading model from {ckpt_path}")
    model = LitModel.load_from_checkpoint(str(ckpt_path))

    print(f"Generating predictions for {split} split...")
    predictions_df = predict_model_as_df(model=model, ckpt_path=ckpt_path)

    return predictions_df


def calculate_mae(predictions_df: pl.DataFrame) -> float:
    """Calculate Mean Absolute Error for yards gained prediction."""
    return (predictions_df["yards_gained"] - predictions_df["expected_yards"]).abs().mean()


def calculate_metrics(predictions_df: pl.DataFrame) -> dict:
    """Calculate comprehensive performance metrics."""
    mae = calculate_mae(predictions_df)

    # Calculate MAE by frame difference (frames before tackle)
    mae_by_frame = (
        predictions_df
        .group_by("frame_diff")
        .agg([
            (pl.col("yards_gained") - pl.col("expected_yards")).abs().mean().alias("mae"),
            pl.count().alias("count"),
        ])
        .sort("frame_diff")
    )

    # Calculate RMSE
    rmse = np.sqrt(((predictions_df["yards_gained"] - predictions_df["expected_yards"]) ** 2).mean())

    # Calculate accuracy within X yards
    within_1_yard = ((predictions_df["yards_gained"] - predictions_df["expected_yards"]).abs() <= 1).mean()
    within_3_yards = ((predictions_df["yards_gained"] - predictions_df["expected_yards"]).abs() <= 3).mean()
    within_5_yards = ((predictions_df["yards_gained"] - predictions_df["expected_yards"]).abs() <= 5).mean()

    return {
        "mae": mae,
        "rmse": rmse,
        "within_1_yard": within_1_yard,
        "within_3_yards": within_3_yards,
        "within_5_yards": within_5_yards,
        "mae_by_frame": mae_by_frame,
    }


def plot_comparison(results: dict):
    """Generate comparison plots."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Overall MAE comparison
    ax = axes[0, 0]
    models = list(results.keys())
    maes = [results[m]["metrics"]["mae"] for m in models]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    ax.bar(models, maes, color=colors)
    ax.set_ylabel("Mean Absolute Error (yards)")
    ax.set_title("Overall MAE Comparison")
    ax.set_ylim(0, max(maes) * 1.2)
    for i, mae in enumerate(maes):
        ax.text(i, mae + 0.05, f'{mae:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: RMSE comparison
    ax = axes[0, 1]
    rmses = [results[m]["metrics"]["rmse"] for m in models]
    ax.bar(models, rmses, color=colors)
    ax.set_ylabel("Root Mean Squared Error (yards)")
    ax.set_title("RMSE Comparison")
    ax.set_ylim(0, max(rmses) * 1.2)
    for i, rmse in enumerate(rmses):
        ax.text(i, rmse + 0.05, f'{rmse:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Accuracy within X yards
    ax = axes[1, 0]
    x = np.arange(len(models))
    width = 0.25
    within_1 = [results[m]["metrics"]["within_1_yard"] for m in models]
    within_3 = [results[m]["metrics"]["within_3_yards"] for m in models]
    within_5 = [results[m]["metrics"]["within_5_yards"] for m in models]

    ax.bar(x - width, within_1, width, label='Within 1 yard', color='#3498db')
    ax.bar(x, within_3, width, label='Within 3 yards', color='#e74c3c')
    ax.bar(x + width, within_5, width, label='Within 5 yards', color='#2ecc71')

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Prediction Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Plot 4: MAE by frame difference
    ax = axes[1, 1]
    for model_name, color in zip(models, colors):
        mae_by_frame = results[model_name]["metrics"]["mae_by_frame"]
        ax.plot(mae_by_frame["frame_diff"], mae_by_frame["mae"],
                label=model_name, linewidth=2, marker='o', color=color)

    ax.set_xlabel("Frames Before Tackle")
    ax.set_ylabel("Mean Absolute Error (yards)")
    ax.set_title("MAE vs. Time to Tackle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "2week_vs_full_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {RESULTS_DIR / '2week_vs_full_comparison.png'}")


def main():
    print("="*60)
    print("2-Week vs Full 18-Week Model Comparison")
    print("="*60)

    results = {}

    # Find and evaluate 2-week Zoo model
    print("\n--- 2-Week Zoo Model ---")
    zoo_2week_ckpt = find_best_model(MODELS_2WEEK_PATH, "zoo")
    if zoo_2week_ckpt:
        zoo_pred = load_model_and_predict(zoo_2week_ckpt, DATA_2WEEK_PATH)
        zoo_metrics = calculate_metrics(zoo_pred)
        results["2-Week Zoo"] = {
            "checkpoint": str(zoo_2week_ckpt),
            "metrics": zoo_metrics,
        }
        print(f"MAE: {zoo_metrics['mae']:.4f} yards")
    else:
        print("No 2-week Zoo model found")

    # Find and evaluate 2-week Transformer model
    print("\n--- 2-Week Transformer Model ---")
    transformer_2week_ckpt = find_best_model(MODELS_2WEEK_PATH, "transformer")
    if transformer_2week_ckpt:
        transformer_2week_pred = load_model_and_predict(transformer_2week_ckpt, DATA_2WEEK_PATH)
        transformer_2week_metrics = calculate_metrics(transformer_2week_pred)
        results["2-Week Transformer"] = {
            "checkpoint": str(transformer_2week_ckpt),
            "metrics": transformer_2week_metrics,
        }
        print(f"MAE: {transformer_2week_metrics['mae']:.4f} yards")
    else:
        print("No 2-week Transformer model found")

    # Find and evaluate 18-week Transformer model
    print("\n--- 18-Week Transformer Model ---")
    transformer_full_ckpt = find_best_model(MODELS_FULL_PATH, "transformer")
    if transformer_full_ckpt:
        transformer_full_pred = load_model_and_predict(transformer_full_ckpt, DATA_FULL_PATH)
        transformer_full_metrics = calculate_metrics(transformer_full_pred)
        results["18-Week Transformer"] = {
            "checkpoint": str(transformer_full_ckpt),
            "metrics": transformer_full_metrics,
        }
        print(f"MAE: {transformer_full_metrics['mae']:.4f} yards")
    else:
        print("No 18-week Transformer model found")

    # Generate summary table
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'Within 3yd':<12}")
    print("-"*60)
    for model_name, data in results.items():
        metrics = data["metrics"]
        print(f"{model_name:<25} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} {metrics['within_3_yards']:<12.2%}")

    # Save results to JSON
    # Convert Polars DataFrames to dict for JSON serialization
    results_serializable = {}
    for model_name, data in results.items():
        results_serializable[model_name] = {
            "checkpoint": data["checkpoint"],
            "mae": data["metrics"]["mae"],
            "rmse": data["metrics"]["rmse"],
            "within_1_yard": data["metrics"]["within_1_yard"],
            "within_3_yards": data["metrics"]["within_3_yards"],
            "within_5_yards": data["metrics"]["within_5_yards"],
        }

    with open(RESULTS_DIR / "2week_vs_full_comparison.json", "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / '2week_vs_full_comparison.json'}")

    # Generate plots
    if len(results) > 0:
        plot_comparison(results)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    import re
    main()
