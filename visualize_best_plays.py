"""
Find and visualize 5 plays of each type with lowest MAE from the 23-entity-football model.
Saves visualizations to ./visualizations/ folder.
"""

import re
from pathlib import Path
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns

sns.set_style('whitegrid')

# Configuration
OUTPUT_DIR = Path('./visualizations')
OUTPUT_DIR.mkdir(exist_ok=True)

# Paths (adjust for your environment)
MODELS_DIR = Path('/content/drive/MyDrive/SportsTrackingTransformer/models_norm_football')
NGS_DATA_DIR = Path('/content/drive/MyDrive/NGS/NFL/REG/')

PLAY_TYPES = ['play_type_pass', 'play_type_rush', 'play_type_sack']
MIN_YARDS = -10

print("="*80)
print("🏈 VISUALIZING BEST PREDICTIONS FROM 23-ENTITY-FOOTBALL MODEL")
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

# Filter to test set
split_col = 'dataset_split' if 'dataset_split' in results_df.columns else 'split'
test_df = results_df.filter(pl.col(split_col) == 'test')
print(f"✓ Test set: {len(test_df):,} predictions")

print("\nStep 3: Loading NGS metadata...")
parquet_files = []
for week_dir in NGS_DATA_DIR.glob("Week*"):
    if week_dir.is_dir():
        parquet_files.extend([f for f in week_dir.glob("*.parquet") if "_mirror" not in f.name])

print(f"Found {len(parquet_files)} parquet files")
ngs_dfs = []
for i, f in enumerate(parquet_files[:100]):
    if i % 25 == 0:
        print(f"  Loading file {i+1}/100...")
    ngs_dfs.append(pl.read_parquet(f))

ngs_df = pl.concat(ngs_dfs, how="vertical_relaxed")

# Normalize column names
rename_map = {'gamekey': 'gameId', 'playid': 'playId', 'frame_id': 'frameId'}
for old, new in rename_map.items():
    if old in ngs_df.columns:
        ngs_df = ngs_df.rename({old: new})

ngs_metadata = ngs_df.select(['gameId', 'playId', 'frameId', 'play_type', 'event']).unique()
print(f"✓ Loaded {len(ngs_metadata):,} metadata records")

print("\nStep 4: Joining predictions with metadata...")
# Handle mirrored plays
test_orig = test_df.filter(pl.col('mirrored') == False)
joined_orig = test_orig.join(ngs_metadata, on=['gameId', 'playId', 'frameId'], how='left')

test_mirror = test_df.filter(pl.col('mirrored') == True)
if len(test_mirror) > 0:
    play_meta = ngs_metadata.select(['gameId', 'playId', 'play_type']).unique()
    joined_mirror = test_mirror.join(play_meta, on=['gameId', 'playId'], how='left')
    joined = pl.concat([joined_orig, joined_mirror])
else:
    joined = joined_orig

print(f"✓ Joined: {len(joined):,} predictions with metadata")

print("\nStep 5: Computing MAE per play...")
# Compute MAE for each play
joined = joined.with_columns([
    (pl.col('yards_gained_class') + MIN_YARDS).alias('true_yards'),
    (pl.col('predicted_class') + MIN_YARDS).alias('pred_yards')
])

joined = joined.with_columns(
    (pl.col('true_yards') - pl.col('pred_yards')).abs().alias('abs_error')
)

# Get play-level MAE (average across all frames in the play)
play_mae = joined.group_by(['gameId', 'playId', 'play_type']).agg([
    pl.col('abs_error').mean().alias('play_mae'),
    pl.col('true_yards').first().alias('yards_gained'),
    pl.col('mirrored').first().alias('mirrored')
])

print(f"✓ Computed MAE for {len(play_mae):,} plays")

print("\nStep 6: Finding best plays for each type...")
best_plays = []

for play_type in PLAY_TYPES:
    play_type_clean = play_type.replace('play_type_', '').capitalize()
    subset = play_mae.filter(pl.col('play_type') == play_type).sort('play_mae')

    if len(subset) >= 5:
        top5 = subset.head(5)
        best_plays.append(top5)
        print(f"\n{play_type_clean}:")
        for i, row in enumerate(top5.iter_rows(named=True), 1):
            print(f"  {i}. Game {row['gameId']}, Play {row['playId']}: "
                  f"MAE={row['play_mae']:.2f}y, Actual={row['yards_gained']}y")
    else:
        print(f"\n{play_type_clean}: Only {len(subset)} plays found (need 5)")

if not best_plays:
    print("\n❌ No plays found to visualize!")
    exit(1)

best_plays_df = pl.concat(best_plays)
print(f"\n✓ Selected {len(best_plays_df)} plays total")

print("\nStep 7: Creating visualizations...")

def plot_play_trajectory(game_id, play_id, ngs_data, predictions, output_path):
    """Create a static plot showing play trajectory with predictions."""

    # Get play data
    play_ngs = ngs_data.filter(
        (pl.col('gameId') == game_id) & (pl.col('playId') == play_id)
    ).sort('frameId')

    play_preds = predictions.filter(
        (pl.col('gameId') == game_id) & (pl.col('playId') == play_id)
    ).sort('frameId')

    if len(play_ngs) == 0 or len(play_preds) == 0:
        print(f"  ⚠️  No data for game {game_id}, play {play_id}")
        return

    # Get play info
    play_type = play_preds['play_type'][0] if 'play_type' in play_preds.columns else 'unknown'
    true_yards = play_preds['true_yards'][0]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Top panel: Field view with player trajectories
    ax_field = axes[0]
    ax_field.set_xlim(0, 120)
    ax_field.set_ylim(0, 53.3)
    ax_field.set_aspect('equal')

    # Draw field
    ax_field.add_patch(patches.Rectangle((0, 0), 120, 53.3, fill=False, edgecolor='white', linewidth=2))
    for yard in range(10, 110, 10):
        ax_field.axvline(yard, color='white', alpha=0.3, linewidth=0.5)

    ax_field.set_facecolor('#2e7d32')
    ax_field.set_title(f'Game {game_id}, Play {play_id} - {play_type} ({true_yards}y)',
                       fontsize=14, fontweight='bold')

    # Plot player trajectories (simplified - would need actual x,y data)
    ax_field.text(60, 26.65, 'Field View Not Available\n(Requires player position data)',
                  ha='center', va='center', fontsize=16, color='white', alpha=0.7)

    # Bottom panel: Predictions over time
    ax_pred = axes[1]

    frame_ids = play_preds['frameId'].to_list()
    pred_yards = play_preds['pred_yards'].to_list()

    ax_pred.plot(frame_ids, pred_yards, 'o-', linewidth=2, markersize=6,
                 label='Predicted Yards', color='#1f77b4')
    ax_pred.axhline(true_yards, color='red', linestyle='--', linewidth=2,
                    label=f'Actual Yards: {true_yards}')

    ax_pred.set_xlabel('Frame ID', fontsize=12)
    ax_pred.set_ylabel('Predicted Yards', fontsize=12)
    ax_pred.set_title('Prediction Evolution Throughout Play', fontsize=14, fontweight='bold')
    ax_pred.legend(fontsize=10)
    ax_pred.grid(True, alpha=0.3)

    # Add error annotation
    final_pred = pred_yards[-1] if pred_yards else 0
    mae = abs(final_pred - true_yards)
    ax_pred.text(0.98, 0.02, f'Final MAE: {mae:.2f} yards',
                 transform=ax_pred.transAxes, ha='right', va='bottom',
                 fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")

# Create visualizations for each selected play
for row in best_plays_df.iter_rows(named=True):
    game_id = row['gameId']
    play_id = row['playId']
    play_type = row['play_type'].replace('play_type_', '')
    mae = row['play_mae']

    output_filename = f"{play_type}_game{game_id}_play{play_id}_mae{mae:.2f}.png"
    output_path = OUTPUT_DIR / output_filename

    plot_play_trajectory(game_id, play_id, ngs_df, joined, output_path)

print("\n" + "="*80)
print("✅ VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nAll visualizations saved to: {OUTPUT_DIR.absolute()}")
print(f"Total files created: {len(list(OUTPUT_DIR.glob('*.png')))}")
