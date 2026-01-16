"""
Join model predictions with NGS tracking data.
Loads predictions from best model and joins with one week of NGS data.
Saves enriched data with predictions to CSV file.
"""

import re
from pathlib import Path
import polars as pl

# Configuration
MODELS_DIR = Path('/content/drive/MyDrive/SportsTrackingTransformer/models_norm_football')
NGS_DATA_DIR = Path('/content/drive/MyDrive/NGS/NFL/REG/')
OUTPUT_FILE = Path('./predictions_with_ngs_data.csv')

MIN_YARDS = -10

print("="*80)
print("🏈 JOINING PREDICTIONS WITH NGS TRACKING DATA")
print("="*80)
print()

def get_epoch_val_loss_from_ckpt(ckpt_path: Path):
    """Extract epoch and val_loss from checkpoint filename."""
    val_loss_pattern = re.compile(r"epoch=(\d+)-val_loss=([\d\.]+)")
    match = val_loss_pattern.search(ckpt_path.name)
    if match:
        try:
            return int(match.group(1)), float(match.group(2).rstrip("."))
        except:
            return -1, float('inf')
    return -1, float('inf')

def find_best_model(models_root: Path):
    """Find the best checkpoint by lowest val_loss."""
    model_dir = models_root / 'transformer'
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return None

    best_ckpt = None
    best_val_loss = float('inf')

    for config_dir in model_dir.iterdir():
        if not config_dir.is_dir() or config_dir.name == 'best_models':
            continue

        ckpt_dir = config_dir / 'checkpoints'
        if not ckpt_dir.exists():
            continue

        ckpts = list(ckpt_dir.glob('*.ckpt'))
        for ckpt in ckpts:
            _, val_loss = get_epoch_val_loss_from_ckpt(ckpt)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = ckpt

    if best_ckpt:
        results_path = best_ckpt.with_suffix('.results.parquet')
        if results_path.exists():
            return results_path

    return None

print("Step 1: Finding best model results...")
results_path = find_best_model(MODELS_DIR)

if not results_path:
    print("❌ No results file found!")
    exit(1)

print(f"✓ Found results: {results_path.name}")

print("\nStep 2: Loading results...")
results_df = pl.read_parquet(results_path)
print(f"✓ Loaded {len(results_df):,} predictions")

# Filter to test set and non-mirrored plays only
split_col = 'dataset_split' if 'dataset_split' in results_df.columns else 'split'
test_df = results_df.filter(
    (pl.col(split_col) == 'test') &
    (pl.col('mirrored') == False)
)
print(f"✓ Test set (non-mirrored): {len(test_df):,} predictions")

print("\nStep 3: Loading NGS data (single file sample)...")
# Load just one parquet file as a sample
all_parquet_files = []
for week_dir in NGS_DATA_DIR.glob("Week*"):
    if week_dir.is_dir():
        all_parquet_files.extend([f for f in week_dir.glob("*.parquet") if "_mirror" not in f.name])

if not all_parquet_files:
    print("❌ No parquet files found!")
    exit(1)

# Use the first parquet file as sample
sample_file = all_parquet_files[0]
print(f"Loading single file: {sample_file.parent.name}/{sample_file.name}")

ngs_df = pl.read_parquet(sample_file)
print(f"✓ Loaded {len(ngs_df):,} NGS tracking records from single file")

# Normalize column names
rename_map = {'gamekey': 'gameId', 'playid': 'playId', 'frame_id': 'frameId'}
for old, new in rename_map.items():
    if old in ngs_df.columns:
        ngs_df = ngs_df.rename({old: new})

print(f"NGS columns: {ngs_df.columns}")

print("\nStep 4: Joining predictions with NGS data...")
# Add true_yards to predictions
predictions = test_df.with_columns([
    (pl.col('yards_gained_class') + MIN_YARDS).alias('true_yards')
])

# Select key prediction columns to join
pred_cols = ['gameId', 'playId', 'frameId', 'expected_yards', 'predicted_class',
             'true_yards', 'yards_gained_class']

predictions_slim = predictions.select([c for c in pred_cols if c in predictions.columns])

# Join NGS data with predictions
joined = ngs_df.join(
    predictions_slim,
    on=['gameId', 'playId', 'frameId'],
    how='inner'  # Only keep records that have both NGS data and predictions
)

print(f"✓ Joined: {len(joined):,} records")

# Add computed columns
if 'expected_yards' in joined.columns and 'true_yards' in joined.columns:
    joined = joined.with_columns([
        (pl.col('true_yards') - pl.col('expected_yards')).abs().alias('prediction_error'),
        (pl.col('true_yards') - pl.col('expected_yards')).alias('prediction_residual')
    ])

print("\nStep 5: Saving enriched data...")
joined.write_csv(OUTPUT_FILE)
print(f"✓ Saved to: {OUTPUT_FILE}")

print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print(f"Input predictions: {len(test_df):,}")
print(f"Input NGS records: {len(ngs_df):,}")
print(f"Output joined records: {len(joined):,}")
print(f"\nColumns in output: {len(joined.columns)}")
print("Prediction columns added:")
print("  - expected_yards (weighted average from softmax)")
print("  - predicted_class (argmax class)")
print("  - true_yards (actual yards gained)")
print("  - prediction_error (absolute error)")
print("  - prediction_residual (signed error)")
print(f"\nSaved to: {OUTPUT_FILE.absolute()}")
