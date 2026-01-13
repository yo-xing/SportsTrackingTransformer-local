#!/usr/bin/env python3
"""
Run the full pipeline for transformer model only (skips zoo model training).
This is memory-efficient and suitable for running on full 18-week dataset.

Usage:
    python run_transformer_only.py [--force] [--skip-prep] [--skip-precompute]

Options:
    --force            Force recompute all stages (ignore cache)
    --skip-prep        Skip data preparation stage (use existing data)
    --skip-precompute  Skip feature precomputation stage (use existing features)
"""

import subprocess
import sys
from argparse import ArgumentParser


def run_command(cmd: str, description: str):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✓ {description} complete\n")


def main():
    parser = ArgumentParser(description="Run transformer-only training pipeline")
    parser.add_argument("--force", action="store_true", help="Force recompute all stages")
    parser.add_argument("--skip-prep", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip feature precomputation")
    parser.add_argument("--sample", type=float, default=None, help="Sample fraction of data (e.g., 0.1 for 10%%)")
    args = parser.parse_args()

    force_flag = "--force" if args.force else ""
    sample_flag = f"--sample {args.sample}" if args.sample else ""

    print("\n" + "="*60)
    print("Transformer-Only Training Pipeline")
    print("="*60 + "\n")

    # Use separate cache directories for full 18-week data
    # This prevents overwriting the 2-week cache used by regular DVC pipeline
    drive_cache_prep = "/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_full18weeks"
    drive_cache_datasets = "/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_full18weeks"
    local_output_prep = "data/split_prepped_data_extra_full18weeks"
    local_output_datasets = "data/datasets_extra_full18weeks"
    local_output_models = "models_full18weeks"

    # Stage 1: Prepare extra data (with all 18 weeks)
    if not args.skip_prep:
        # Use all 18 weeks for full dataset training
        all_weeks = " ".join([f"{i:02d}" for i in range(1, 19)])
        sample_desc = f" ({args.sample*100:.0f}% sample)" if args.sample else ""
        run_command(
            f"uv run python src/prep_extra_data.py --weeks {all_weeks} "
            f"--output-dir {local_output_prep} --drive-dir {drive_cache_prep} {sample_flag}",
            f"Stage 1/3: Preparing extra data (18 weeks{sample_desc})"
        )
    else:
        print("\n⏭️  Skipping data preparation stage\n")

    # Stage 2: Precompute datasets (transformer only, skip zoo)
    if not args.skip_precompute:
        run_command(
            f"uv run python src/datasets.py "
            f"--prepped-data-dir {local_output_prep} "
            f"--dataset-dir {local_output_datasets} "
            f"--drive-dir {drive_cache_datasets} "
            f"--model-types transformer",
            "Stage 2/3: Precomputing feature transforms (transformer only)"
        )
    else:
        print("\n⏭️  Skipping feature precomputation stage\n")

    # Stage 3: Train transformer models only
    # Optimized for A100 with 167GB RAM: batch_size=256, num_workers=16
    # Note: --skip-existing removed to allow retraining models with NaN loss
    run_command(
        f"uv run python src/train.py --model_type transformer --device 0 "
        f"--prepped-data-dir {local_output_prep} "
        f"--dataset-dir {local_output_datasets} --models-dir {local_output_models} "
        f"--batch-size 256 --num-workers 16",
        "Stage 3/3: Training transformer models"
    )

    # Stage 4: Backup trained models to Google Drive
    drive_models_dir = "/content/drive/MyDrive/SportsTrackingTransformer/models_full18weeks"
    from pathlib import Path
    if Path("/content/drive/MyDrive").exists():
        print("\n" + "="*60)
        print("Stage 4/4: Backing up models to Google Drive")
        print("="*60 + "\n")

        Path(drive_models_dir).mkdir(exist_ok=True, parents=True)

        local_transformer_dir = Path(local_output_models) / "transformer"
        if local_transformer_dir.exists():
            import subprocess
            print(f"Copying models to {drive_models_dir}/transformer/")
            subprocess.run(
                ["rsync", "-av", "--progress", f"{local_transformer_dir}/", f"{drive_models_dir}/transformer/"],
                check=True
            )
            print("✓ Models backed up to Google Drive\n")
        else:
            print(f"⚠️  No transformer models found at {local_transformer_dir}\n")
    else:
        print("\nGoogle Drive not mounted, skipping model backup\n")

    print("\n" + "="*60)
    print("Pipeline Complete! 🎉")
    print("="*60 + "\n")
    print("Trained models are in:")
    print(f"  - Local: {local_output_models}/transformer/")
    if Path("/content/drive/MyDrive").exists():
        print(f"  - Google Drive: {drive_models_dir}/transformer/")
    print("\nCache locations:")
    print(f"  - Prep data cache: {drive_cache_prep}")
    print(f"  - Dataset cache: {drive_cache_datasets}")
    print(f"  - Local prep data: {local_output_prep}")
    print(f"  - Local datasets: {local_output_datasets}\n")
    print("Note: These are separate from the 2-week DVC pipeline caches.\n")
    print("Next steps:")
    print("  - To compare models: python compare_2week_vs_full.py\n")


if __name__ == "__main__":
    main()
