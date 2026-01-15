# Branch: game-state-norm

## Overview
This branch is a clean clone of `add-game-state-features` with only cache locations and the `--use-gamestate-prep` flag changed.

## Features
- **11 features**: 6 player features + 5 game state features
  - Player: x_rel, y_rel, vx, vy, side, is_ball_carrier
  - Game state: yardsToGo, down, distanceToGoal, quarter, half_seconds_remaining
- **23 entities**: 22 players + 1 football
- **Normalized game state features**: All game state features normalized to [0, 1] range

## Normalization Details
All game state features are normalized during dataset precomputation (Stage 2):

1. **yardsToGo**: Clip to 40 yards, normalize by dividing by 40 → [0, 1]
2. **down**: Normalize (1-4) using `(x - 1) / 3` → [0, 1]
3. **distanceToGoal**: Normalize (0-100 yards) by dividing by 100 → [0, 1]
4. **quarter**: Normalize (1-4) using `(x - 1) / 3` → [0, 1]
5. **half_seconds_remaining**: Normalize (0-1800 seconds) by dividing by 1800 → [0, 1]

## Cache Locations

### Google Drive Caches
- **Prep data**: `/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_gamestate_norm`
- **Datasets**: `/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_gamestate_norm`
- **Models**: `/content/drive/MyDrive/SportsTrackingTransformer/models_gamestate_norm`

### Local Caches
- **Prep data**: `data/split_prepped_data_extra_gamestate_norm/`
- **Datasets**: `data/datasets_extra_gamestate_norm/`
- **Models**: `models_gamestate_norm/`

## New Feature: --use-gamestate-prep Flag

Added to `run_transformer_only.py` to allow using preprocessed data from the `add-game-state-features` branch:

```bash
# Use prep data from add-game-state-features branch (skip Stage 1)
python run_transformer_only.py --use-gamestate-prep
```

When this flag is set:
- Stage 1 (data preparation) is automatically skipped
- Uses prep cache: `/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_gamestate_23`
- Dataset cache remains unique: `NewDataSportsTrackingTransformer_cache_gamestate_norm`

## Differences from add-game-state-features Branch

| Aspect | add-game-state-features | game-state-norm |
|--------|------------------------|-----------------|
| **Features** | 11 (6 player + 5 game state) | 11 (6 player + 5 game state) ✓ |
| **Entities** | 23 (22 players + 1 football) | 23 (22 players + 1 football) ✓ |
| **Normalization** | Yes (all 5 game state) | Yes (all 5 game state) ✓ |
| **Prep cache** | `_gamestate_23` | `_gamestate_norm` |
| **Dataset cache** | `_gamestate_23` | `_gamestate_norm` |
| **Model cache** | `models` | `models_gamestate_norm` |
| **--use-gamestate-prep flag** | No | Yes ✓ |

## Usage

### Full Pipeline (with prep)
```bash
python run_transformer_only.py
```

### Using add-game-state-features Prep Data (skip Stage 1)
```bash
python run_transformer_only.py --use-gamestate-prep
```

### Other Options
```bash
# Force recompute all stages
python run_transformer_only.py --force

# Skip specific stages
python run_transformer_only.py --skip-prep --skip-precompute

# Train with specific settings
python run_transformer_only.py --device 0 --patience 15 --batch-size 256
```

## Pipeline Stages

1. **Stage 1: Data Preparation** - Preprocess raw NFL tracking data (can skip with `--use-gamestate-prep`)
2. **Stage 2: Dataset Precomputation** - Create pickled datasets with normalized features
3. **Stage 3: Model Training** - Train transformer models
4. **Stage 4: Models to Drive** - Sync models to Google Drive
5. **Stage 5: Results Summary** - Generate results summary

## Files Changed from add-game-state-features

1. `src/prep_extra_data.py` - Updated cache paths
2. `src/datasets.py` - Updated cache paths
3. `src/train.py` - Updated model paths
4. `src/pick_best_models.py` - Updated model paths
5. `src/generate_results_summary.py` - Updated example command
6. `run_transformer_only.py` - Added `--use-gamestate-prep` flag, updated all cache paths
