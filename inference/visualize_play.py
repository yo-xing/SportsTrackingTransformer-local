"""
🏈 FINAL CORRECTED VERSION: Cross-Branch Model Comparison
Copy this entire cell into Colab and run it!

CORRECTED MODEL DESCRIPTIONS:
All models use 18 weeks of data. Differences are in FEATURES and ENTITY TRACKING.
"""

import re
from pathlib import Path
from typing import Dict, Tuple
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score, top_k_accuracy_score

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 12)

print("="*80)
print("🏈 NFL YARDS PREDICTION: FEATURE ENGINEERING COMPARISON")
print("="*80)
print("""
⚠️  All models use the SAME 18 weeks of training data!
   Differences are FEATURES and ENTITY TRACKING only, not data volume.

WHAT ARE WE PREDICTING?
  • Yards gained on each play (from -10 to +99 yards)
  • 110-class classification problem (classes 0-109)
  • Models predict at EVERY frame as play unfolds

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL COMPARISON TABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────────────┬──────────────┬───────────────┬─────────────────────┐
│ Feature              │ 22-entity    │ game-state    │ full-data-branch    │
│                      │ -norm        │ -norm         │ -yo                 │
├──────────────────────┼──────────────┼───────────────┼─────────────────────┤
│ Entities Tracked     │ 22 players   │ 22 + football │ 22 players          │
│ Total Features       │ 7            │ 11            │ 6                   │
│ Player Features      │ 6            │ 6             │ 6                   │
│ Game State Features  │ 1 (goal)     │ 5 (full)      │ 0                   │
│ Football Tracking    │ NO           │ YES (23rd)    │ NO                  │
│ Coordinate System    │ Normalized   │ Normalized    │ Normalized          │
│ Training Data        │ 18 weeks     │ 18 weeks      │ 18 weeks            │
│ Design Philosophy    │ Minimalist+  │ Contextual    │ Baseline            │
└──────────────────────┴──────────────┴───────────────┴─────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODELS BEING COMPARED:

1️⃣  22-ENTITY-NORM (7 features, 22 players, NO football)
   Player Features (6): x_rel, y_rel, vx, vy, side, is_ball_carrier
   Game State (1): distanceToGoal
   Design: Minimalist with basic game state (distance to goal only)

2️⃣  GAME-STATE-NORM (11 features, 23 entities = 22 players + 1 football)
   Player Features (6): x_rel, y_rel, vx, vy, side, is_ball_carrier
   Game State (5): yardsToGo, down, distanceToGoal, quarter, half_seconds_remaining
   Design: Contextual - includes football tracking + full game state

3️⃣  FULL-DATA-BRANCH-YO (6 features, 22 players, NO football)
   Features (6): x_rel, y_rel, vx, vy, side, is_ball_carrier
   Design: Baseline - minimal feature engineering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY FEATURE EXPLANATIONS:
  • x_rel, y_rel: Player position relative to ball carrier (normalized)
  • vx, vy: Player velocity in x and y directions (normalized)
  • side: Team indicator (offense=1, defense=-1, football=0)
  • is_ball_carrier: Binary flag (1=ball carrier, 0=other players)
  • distanceToGoal: Yards from goal line
  • yardsToGo: Yards needed for first down
  • down: Current down (1st/2nd/3rd/4th)
  • quarter: Current quarter (1-4)
  • half_seconds_remaining: Time left in current half

EVALUATION METRIC - MAE (Mean Absolute Error):
  • MAE measures average yards prediction error
  • Example: If true yards = +5 and predicted yards = +8, error = 3 yards
  • Lower MAE = better accuracy in yards prediction
  • Complements cross-entropy loss (which measures probability distribution quality)
  • More interpretable: "Model is off by X yards on average"

RESEARCH QUESTIONS:
  1. Does tracking the FOOTBALL (23rd entity) improve predictions?
  2. Do GAME STATE features (yardsToGo, down, quarter, time) help?
  3. Does minimal game state (just distanceToGoal) vs full context matter?
  4. Which model is best for passes? For rushes? For sacks?
  5. Do predictions improve as plays progress (early → middle → late)?
  6. Are models overfitting or generalizing well (train vs val vs test)?
""")

# Configuration
NGS_DATA_DIR = Path('/content/drive/MyDrive/NGS/NFL/REG/')
BRANCH_CONFIGS = {
    '23-entity-norm': {
        'gdrive_models': Path('/content/drive/MyDrive/SportsTrackingTransformer/models_norm'),
        'color': '#1f77b4',
        'label': '22-entity-norm\n(7 feat, 22 ents, +goal)',
        'display_name': '22-entity-norm'
    },
    'game-state-norm': {
        'gdrive_models': Path('/content/drive/MyDrive/SportsTrackingTransformer/models_gamestate_norm'),
        'color': '#ff7f0e',
        'label': 'game-state-norm\n(11 feat, 23 ents, +context)',
        'display_name': 'game-state-norm'
    },
    'full-data-branch-yo': {
        'gdrive_models': Path('/content/drive/MyDrive/SportsTrackingTransformer/models'),
        'color': '#2ca02c',
        'label': 'full-data-branch-yo\n(6 feat, 22 ents, baseline)',
        'display_name': 'full-data-branch-yo'
    }
}

KEY_EVENTS = ['ball_snap', 'handoff', 'pass_forward', 'pass_arrived',
              'first_contact', 'tackle', 'out_of_bounds', 'touchdown']
PLAY_TYPES = ['play_type_pass', 'play_type_rush', 'play_type_sack']

def get_epoch_val_loss_from_ckpt(ckpt_path: Path) -> Tuple[int, float]:
    """Extract epoch and val_loss from checkpoint filename."""
    val_loss_pattern = re.compile(r"epoch=(\d+)-val_loss=([\d\.]+)")
    match = val_loss_pattern.search(ckpt_path.name)
    if match:
        try:
            return int(match.group(1)), float(match.group(2).rstrip("."))
        except:
            return -1, float('inf')
    return -1, float('inf')

def find_best_models(models_root: Path):
    """
    Find best model by:
    1. Scan transformer directory for all configs (M32_L4_LR1e-04, etc)
    2. For each config, find checkpoint with lowest val_loss
    3. Return dict of config -> (checkpoint, results, val_loss)
    """
    model_dir = models_root / 'transformer'
    if not model_dir.exists():
        return {}

    best_models = {}
    for config_dir in model_dir.iterdir():
        if not config_dir.is_dir() or config_dir.name == 'best_models':
            continue

        ckpt_dir = config_dir / 'checkpoints'
        if not ckpt_dir.exists():
            continue

        ckpts = list(ckpt_dir.glob('*.ckpt'))
        if ckpts:
            # Find checkpoint with lowest val_loss for this config
            best_ckpt = min(ckpts, key=lambda x: get_epoch_val_loss_from_ckpt(x)[1])
            _, val_loss = get_epoch_val_loss_from_ckpt(best_ckpt)
            results_path = best_ckpt.with_suffix('.results.parquet')
            best_models[config_dir.name] = (best_ckpt, results_path, val_loss)

    return best_models

def compute_metrics(df: pl.DataFrame):
    """Compute evaluation metrics."""
    if len(df) == 0:
        return {}
    y_true = df['yards_gained_class'].to_numpy()
    y_pred = df['predicted_class'].to_numpy()
    n_classes = 110
    y_pred_probs = np.zeros((len(y_pred), n_classes))
    y_pred_probs[np.arange(len(y_pred)), y_pred] = 1.0
    all_labels = np.arange(110)
    MIN_YARDS = -10
    return {
        'cross_entropy': log_loss(y_true, y_pred_probs, labels=all_labels),
        'accuracy': accuracy_score(y_true, y_pred),
        'top5_accuracy': top_k_accuracy_score(y_true, y_pred_probs, k=5, labels=all_labels),
        'mae_yards': np.mean(np.abs((y_true + MIN_YARDS) - (y_pred + MIN_YARDS))),
        'n_samples': len(df)
    }

print("\n" + "="*80)
print("⚙️  STEP 1: FINDING BEST MODELS (by lowest val_loss)")
print("="*80 + "\n")
print("HOW IT WORKS:")
print("  1. Scan each branch's model directory")
print("  2. For each hyperparameter config, find checkpoint with lowest val_loss")
print("  3. Select overall best config across all hyperparameters")
print("  4. Parse val_loss from filename: epoch=13-val_loss=2.7630.ckpt")
print()

branch_best_models = {}
for branch_name, config in BRANCH_CONFIGS.items():
    gdrive_path = config['gdrive_models']
    display_name = config['display_name']
    print(f"\n{display_name}:")
    print(f"  Scanning: {gdrive_path}")

    if not gdrive_path.exists():
        print(f"  ❌ Path not found")
        continue

    all_models = find_best_models(gdrive_path)
    if not all_models:
        print(f"  ❌ No models found")
        continue

    print(f"  ✓ Found {len(all_models)} hyperparameter configs")

    # Show top 5 configs by val_loss
    sorted_models = sorted(all_models.items(), key=lambda x: x[1][2])
    print(f"\n  Top 5 configs by validation loss:")
    for i, (cfg, (ckpt, res, vl)) in enumerate(sorted_models[:5], 1):
        epoch, _ = get_epoch_val_loss_from_ckpt(ckpt)
        res_exists = '✓' if res.exists() else '✗'
        print(f"    {i}. {cfg:30s} val_loss={vl:.4f} epoch={epoch:3d} results:{res_exists}")

    # Find best model WITH results
    for cfg, (ckpt, res, vl) in sorted_models:
        if res.exists():
            branch_best_models[branch_name] = {
                'config': cfg,
                'checkpoint': ckpt,
                'results': res,
                'val_loss': vl
            }
            print(f"\n  🏆 SELECTED: {cfg}")
            print(f"     Val Loss: {vl:.4f}")
            print(f"     Has results: ✓")
            break

if not branch_best_models:
    print("\n❌ No models with results found!")
    exit()

print("\n" + "="*80)
print("📊 STEP 2: TRAIN vs VAL vs TEST COMPARISON")
print("="*80 + "\n")
print("Evaluating each selected model on all three splits...\n")

split_comparison = []
for branch_name, info in branch_best_models.items():
    results_df = pl.read_parquet(info['results'])
    split_col = 'dataset_split' if 'dataset_split' in results_df.columns else 'split'

    display_name = BRANCH_CONFIGS[branch_name]['display_name']
    print(f"{display_name:25s} ({info['config']})")

    for split_name in ['train', 'val', 'test']:
        split_data = results_df.filter(pl.col(split_col) == split_name)
        if len(split_data) > 0:
            metrics = compute_metrics(split_data)
            metrics['branch'] = display_name
            metrics['split'] = split_name
            split_comparison.append(metrics)
            print(f"  {split_name:5s}: {metrics['n_samples']:7,} samples | "
                  f"Acc={metrics['accuracy']:.3f} | "
                  f"Top5={metrics['top5_accuracy']:.3f} | "
                  f"MAE={metrics['mae_yards']:.2f}y | "
                  f"Loss={metrics['cross_entropy']:.4f}")
        else:
            print(f"  {split_name:5s}: NO DATA")
    print()

split_df = pd.DataFrame(split_comparison)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('📊 Model Performance: Train vs Val vs Test', fontsize=18, fontweight='bold')

for (metric, title, is_pct, lower_better), ax in zip([
    ('cross_entropy', 'Cross-Entropy Loss', False, True),
    ('accuracy', 'Accuracy (%)', True, False),
    ('top5_accuracy', 'Top-5 Accuracy (%)', True, False),
    ('mae_yards', 'Mean Absolute Error (yards)', False, True)
], axes.flat):
    for branch_key in BRANCH_CONFIGS.keys():
        display_name = BRANCH_CONFIGS[branch_key]['display_name']
        if display_name in split_df['branch'].values:
            data = split_df[split_df['branch'] == display_name]
            vals = data[metric].values
            if is_pct:
                vals *= 100
            ax.plot(['Train', 'Val', 'Test'], vals, marker='o', linewidth=2, markersize=10,
                   label=display_name, color=BRANCH_CONFIGS[branch_key]['color'])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Dataset Split', fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.legend(title='Model', fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('train_val_test_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: train_val_test_comparison.png\n")
plt.show()

print("-"*80)
print("TRAIN vs VAL vs TEST PERFORMANCE TABLE")
print("-"*80 + "\n")
display_cols = ['branch', 'split', 'accuracy', 'top5_accuracy', 'mae_yards', 'cross_entropy', 'n_samples']
print(split_df[display_cols].to_string(index=False))

print("\n" + "="*80)
print("📂 STEP 3: LOADING NGS DATA")
print("="*80 + "\n")

print("Loading original NGS data to join play_type and event...")
parquet_files = []
for week_dir in NGS_DATA_DIR.glob("Week*"):
    if week_dir.is_dir():
        parquet_files.extend([f for f in week_dir.glob("*.parquet") if "_mirror" not in f.name])

print(f"Found {len(parquet_files)} parquet files")
print(f"Loading first 100 files (adjust if needed)...")

ngs_dfs = []
for i, f in enumerate(parquet_files[:100]):
    if i % 25 == 0:
        print(f"  Loading file {i+1}/100...")
    ngs_dfs.append(pl.read_parquet(f))

ngs_df = pl.concat(ngs_dfs, how="vertical_relaxed")

rename_map = {'gamekey': 'gameId', 'playid': 'playId', 'frame_id': 'frameId'}
for old, new in rename_map.items():
    if old in ngs_df.columns:
        ngs_df = ngs_df.rename({old: new})

ngs_metadata = ngs_df.select(['gameId', 'playId', 'frameId', 'play_type', 'event']).unique()
print(f"✓ Loaded {len(ngs_metadata):,} unique records")

print("\n" + "="*80)
print("🔗 STEP 4: JOINING PREDICTIONS WITH NGS METADATA")
print("="*80 + "\n")

enhanced_results = {}
for branch_name, info in branch_best_models.items():
    results_df = pl.read_parquet(info['results'])
    display_name = BRANCH_CONFIGS[branch_name]['display_name']
    print(f"{display_name}: {len(results_df):,} predictions")

    # Join with NGS (handle mirrored plays)
    results_orig = results_df.filter(pl.col('mirrored') == False)
    joined_orig = results_orig.join(ngs_metadata, on=['gameId', 'playId', 'frameId'], how='left')

    results_mirror = results_df.filter(pl.col('mirrored') == True)
    if len(results_mirror) > 0:
        play_meta = ngs_metadata.select(['gameId', 'playId', 'play_type', 'event']).unique()
        joined_mirror = results_mirror.join(play_meta, on=['gameId', 'playId'], how='left')
        joined = pl.concat([joined_orig, joined_mirror])
    else:
        joined = joined_orig

    # Add frame position
    frame_info = joined.group_by(['gameId', 'playId', 'mirrored']).agg([
        pl.col('frameId').min().alias('min_frame'),
        pl.col('frameId').max().alias('max_frame'),
        pl.col('frameId').n_unique().alias('total_frames')
    ])
    joined = joined.join(frame_info, on=['gameId', 'playId', 'mirrored'], how='left')
    joined = joined.with_columns((pl.col('frameId') - pl.col('min_frame')).alias('frame_index'))
    joined = joined.with_columns([
        pl.when(pl.col('frame_index') < 10).then(pl.lit('first_10'))
        .when(pl.col('frame_index') >= pl.col('total_frames') - 10).then(pl.lit('last_10'))
        .otherwise(pl.lit('middle')).alias('frame_position')
    ])

    # Filter to test set
    split_col = 'dataset_split' if 'dataset_split' in joined.columns else 'split'
    joined = joined.filter(pl.col(split_col) == 'test')

    enhanced_results[display_name] = joined
    print(f"  ✓ Test set: {len(joined):,} predictions with metadata")

print("\n" + "="*80)
print("📊 STEP 5: ANALYZING PERFORMANCE")
print("="*80 + "\n")

# By play type
play_type_results = []
for branch, df in enhanced_results.items():
    for pt in PLAY_TYPES:
        subset = df.filter(pl.col('play_type') == pt)
        if len(subset) > 0:
            m = compute_metrics(subset)
            m['branch'] = branch
            m['play_type'] = pt.replace('play_type_', '').capitalize()
            play_type_results.append(m)

play_type_df = pd.DataFrame(play_type_results)
print(f"✓ Play type analysis: {len(play_type_df)} groups")

# By frame position
frame_pos_results = []
for branch, df in enhanced_results.items():
    for pos in ['first_10', 'middle', 'last_10']:
        subset = df.filter(pl.col('frame_position') == pos)
        if len(subset) > 0:
            m = compute_metrics(subset)
            m['branch'] = branch
            m['frame_position'] = pos
            frame_pos_results.append(m)

frame_pos_df = pd.DataFrame(frame_pos_results)
print(f"✓ Frame position analysis: {len(frame_pos_df)} groups")

# By event
event_results = []
for branch, df in enhanced_results.items():
    for event in KEY_EVENTS:
        subset = df.filter(pl.col('event') == event)
        if len(subset) > 100:
            m = compute_metrics(subset)
            m['branch'] = branch
            m['event'] = event.replace('_', ' ').title()
            event_results.append(m)

event_df = pd.DataFrame(event_results)
print(f"✓ Event analysis: {len(event_df)} groups")

print("\n" + "="*80)
print("📈 STEP 6: CREATING VISUALIZATIONS")
print("="*80 + "\n")

# Plot 1: Play type
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('🏈 Performance by Play Type (Test Set)', fontsize=18, fontweight='bold')

for (metric, title, is_pct), ax in zip([
    ('cross_entropy', 'Cross-Entropy Loss', False),
    ('accuracy', 'Accuracy (%)', True),
    ('top5_accuracy', 'Top-5 Accuracy (%)', True),
    ('mae_yards', 'Mean Absolute Error (yards)', False)
], axes.flat):
    pivot = play_type_df.pivot(index='play_type', columns='branch', values=metric)
    if is_pct:
        pivot *= 100
    pivot.plot(kind='bar', ax=ax, width=0.75, rot=0)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Play Type', fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.legend(title='Model', fontsize=8, loc='best')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('play_type_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: play_type_comparison.png")
plt.show()

# Plot 2: Frame position
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('⏱️  Performance Throughout Play (Test Set)', fontsize=18, fontweight='bold')

position_order = {'first_10': 0, 'middle': 1, 'last_10': 2}
frame_pos_df['order'] = frame_pos_df['frame_position'].map(position_order)
frame_pos_df = frame_pos_df.sort_values(['branch', 'order'])
pos_labels = {'first_10': 'First 10\nFrames', 'middle': 'Middle\nFrames', 'last_10': 'Last 10\nFrames'}

for (metric, title, is_pct), ax in zip([
    ('cross_entropy', 'Cross-Entropy Loss', False),
    ('accuracy', 'Accuracy (%)', True),
    ('top5_accuracy', 'Top-5 Accuracy (%)', True),
    ('mae_yards', 'Mean Absolute Error (yards)', False)
], axes.flat):
    for branch_name in enhanced_results.keys():
        data = frame_pos_df[frame_pos_df['branch'] == branch_name]
        if len(data) > 0:
            vals = data[metric].values
            if is_pct:
                vals *= 100
            x_labels = [pos_labels[p] for p in data['frame_position']]
            # Find the branch_key for this display_name
            branch_key = [k for k, v in BRANCH_CONFIGS.items() if v['display_name'] == branch_name][0]
            ax.plot(x_labels, vals, marker='o', linewidth=2, markersize=10,
                   label=branch_name, color=BRANCH_CONFIGS[branch_key]['color'])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Frame Position', fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.legend(title='Model', fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frame_position_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: frame_position_comparison.png")
plt.show()

# Plot 3: Event heatmap
if len(event_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('🎯 Performance by Event Type (Test Set)', fontsize=18, fontweight='bold')

    for (metric, title), ax in zip([('accuracy', 'Accuracy'), ('mae_yards', 'MAE (yards)')], axes):
        pivot = event_df.pivot(index='event', columns='branch', values=metric)
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn' if metric == 'accuracy' else 'RdYlGn_r',
                   ax=ax, cbar_kws={'label': title})
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Event', fontsize=12)

    plt.tight_layout()
    plt.savefig('event_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: event_heatmap.png")
    plt.show()

print("\n" + "="*80)
print("📋 SUMMARY TABLES")
print("="*80)

print("\n" + "-"*80)
print("PERFORMANCE BY PLAY TYPE")
print("-"*80 + "\n")
print(play_type_df[['branch', 'play_type', 'accuracy', 'top5_accuracy', 'mae_yards', 'n_samples']].to_string(index=False))

print("\n" + "-"*80)
print("PERFORMANCE BY FRAME POSITION")
print("-"*80 + "\n")
print(frame_pos_df[['branch', 'frame_position', 'accuracy', 'top5_accuracy', 'mae_yards', 'n_samples']].to_string(index=False))

if len(event_df) > 0:
    print("\n" + "-"*80)
    print("TOP EVENTS BY SAMPLE SIZE")
    print("-"*80 + "\n")
    top_events = event_df.nlargest(15, 'n_samples')[['branch', 'event', 'accuracy', 'mae_yards', 'n_samples']]
    print(top_events.to_string(index=False))

print("\n" + "="*80)
print("✨ ANALYSIS COMPLETE!")
print("="*80)
print("""
Generated visualizations:
  • train_val_test_comparison.png - Generalization across splits
  • play_type_comparison.png - Pass vs Rush vs Sack
  • frame_position_comparison.png - Early vs Mid vs Late frames
  • event_heatmap.png - Performance across events

CORRECTED KEY FINDINGS TO LOOK FOR:


2. Feature Engineering Impact:
   • 22-entity-norm (7 feat): 6 player features + goal distance, NO football
   • game-state-norm (11 feat): 6 player features + 5 game state + football (23 entities)
   • full-data-branch-yo (6 feat): 6 player features only, NO football


9. MAE Interpretation:
   • Lower MAE = more accurate yards prediction
   • Example: MAE=3.5 means "off by 3.5 yards on average"
   • Compare across models to see which has best yards accuracy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📥 DOWNLOAD RESULTS AS JUPYTER NOTEBOOK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To download this analysis as a Jupyter notebook:

from google.colab import files
import nbformat as nbf

# Create notebook
nb = nbf.v4.new_notebook()
nb['cells'] = [
    nbf.v4.new_markdown_cell("# NFL Yards Prediction: Feature Engineering Comparison\\n\\nAll models use 18 weeks of data."),
    nbf.v4.new_code_cell(__file__)  # This entire script
]

# Save and download
with open('nfl_model_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)

files.download('nfl_model_comparison.ipynb')
print("✓ Notebook downloaded: nfl_model_comparison.ipynb")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Run this in a separate cell AFTER the main analysis completes

from google.colab import files
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Create PDF with all visualizations and tables
pdf_filename = 'nfl_model_comparison_report.pdf'

with PdfPages(pdf_filename) as pdf:
    # Page 1: Title and Model Comparison Table
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.95, 'NFL Yards Prediction: Feature Engineering Comparison',
             ha='center', fontsize=16, weight='bold')
    fig.text(0.5, 0.90, 'All models use 18 weeks of training data',
             ha='center', fontsize=12, style='italic')

    table_text = '''
MODEL COMPARISON TABLE

┌──────────────────────┬──────────────┬───────────────┬─────────────────────┐
│ Feature              │ 22-entity    │ game-state    │ full-data-branch    │
│                      │ -norm        │ -norm         │ -yo                 │
├──────────────────────┼──────────────┼───────────────┼─────────────────────┤
│ Entities Tracked     │ 22 players   │ 22 + football │ 22 players          │
│ Total Features       │ 7            │ 11            │ 6                   │
│ Player Features      │ 6            │ 6             │ 6                   │
│ Game State Features  │ 1 (goal)     │ 5 (full)      │ 0                   │
│ Football Tracking    │ NO           │ YES (23rd)    │ NO                  │
│ Training Data        │ 18 weeks     │ 18 weeks      │ 18 weeks            │
└──────────────────────┴──────────────┴───────────────┴─────────────────────┘

KEY FEATURES:
• 22-entity-norm: 6 player + distanceToGoal (NO football)
• game-state-norm: 6 player + 5 game state + football (23 entities)
• full-data-branch-yo: 6 player features only (NO football)

MAE = Mean Absolute Error (average yards prediction error)
Lower MAE means better accuracy in yards prediction.
    '''
    fig.text(0.1, 0.05, table_text, ha='left', va='bottom',
             fontsize=9, family='monospace')
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 2: Train/Val/Test Performance Table
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.95, 'Train vs Val vs Test Performance',
             ha='center', fontsize=14, weight='bold')

    table_str = split_df[display_cols].to_string(index=False)
    fig.text(0.1, 0.1, table_str, ha='left', va='bottom',
             fontsize=8, family='monospace')
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 3: Train/Val/Test Visualization
    fig = plt.figure(figsize=(11, 8.5))
    img = plt.imread('train_val_test_comparison.png')
    plt.imshow(img)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 4: Play Type Performance
    fig = plt.figure(figsize=(11, 8.5))
    img = plt.imread('play_type_comparison.png')
    plt.imshow(img)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 5: Frame Position Performance
    fig = plt.figure(figsize=(11, 8.5))
    img = plt.imread('frame_position_comparison.png')
    plt.imshow(img)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 6: Event Heatmap (if exists)
    try:
        fig = plt.figure(figsize=(11, 8.5))
        img = plt.imread('event_heatmap.png')
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    except:
        print("  (Event heatmap not found, skipping)")

    # Page 7: Summary Tables
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.95, 'Performance by Play Type',
             ha='center', fontsize=14, weight='bold')

    table_str = play_type_df[['branch', 'play_type', 'accuracy', 'top5_accuracy', 'mae_yards', 'n_samples']].to_string(index=False)
    fig.text(0.1, 0.5, table_str, ha='left', va='top',
             fontsize=8, family='monospace')

    fig.text(0.5, 0.45, 'Performance by Frame Position',
             ha='center', fontsize=14, weight='bold')

    table_str2 = frame_pos_df[['branch', 'frame_position', 'accuracy', 'top5_accuracy', 'mae_yards', 'n_samples']].to_string(index=False)
    fig.text(0.1, 0.05, table_str2, ha='left', va='bottom',
             fontsize=8, family='monospace')
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"\n✅ PDF report created: {pdf_filename}")
files.download(pdf_filename)
print("✅ PDF downloaded to your computer!")