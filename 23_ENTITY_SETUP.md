# 23-Entity Model Setup (Single Game State Feature)

## Branch: 23-entity

This branch is configured to run the transformer model with:
- **Training data**: 18 weeks (60% sample)
- **Features**: 7 total (6 player features + 1 game state feature)
  - Player features: x_rel, y_rel, vx, vy, side, is_ball_carrier
  - Game state: distanceToGoal
- **Entities**: 22 players (no football entity in this branch)
- **Reduced from**: 11 features (removed yardsToGo, down, quarter, half_seconds_remaining)

## Quick Start

```bash
# Using uv (recommended if you have uv installed)
uv run python run_transformer_only.py --use-uv

# Or using system/venv python
./run_transformer_only.py

# Skip filtering if already done
uv run python run_transformer_only.py --use-uv --skip-filter

# Skip training models with existing checkpoints
uv run python run_transformer_only.py --use-uv --skip-existing
```

## Files Modified

1. **src/datasets.py**
   - Updated `transformer_transform_input_frame_df()` to use 7 features
   - Updated `_read_features()` to include distanceToGoal column
   - Features: ["x_rel", "y_rel", "vx", "vy", "side", "is_ball_carrier", "distanceToGoal"]

2. **src/models.py**
   - Updated `feature_len` from 6 to 7 for transformer model (line 306)
   - Added comment explaining the 7 features

3. **filter_features.py** (NEW)
   - Script to filter precomputed datasets from 11 features to 7 features
   - Keeps indices [0, 1, 2, 3, 4, 5, 8] from original 11-feature arrays
   - By default overwrites input datasets in place

4. **run_transformer_only.py** (NEW)
   - Main pipeline script that orchestrates filtering and training
   - Handles the complete workflow with proper flags

## Detailed Usage

### Main Pipeline Script

The `run_transformer_only.py` script is the recommended way to run the complete pipeline:

```bash
# Using uv (recommended - manages dependencies automatically)
uv run python run_transformer_only.py --use-uv

# Or using system/venv python
./run_transformer_only.py

# Skip filtering if datasets are already filtered
uv run python run_transformer_only.py --use-uv --skip-filter

# Skip training models that have existing checkpoints
uv run python run_transformer_only.py --use-uv --skip-existing

# Use a different GPU
uv run python run_transformer_only.py --use-uv --device 1

# Custom early stopping patience
uv run python run_transformer_only.py --use-uv --patience 15

# Combine flags
uv run python run_transformer_only.py --use-uv --skip-filter --skip-existing --device 0
```

**Pipeline flags:**
- `--use-uv`: Use uv for running subprocess commands (required when using `uv run python`)
- `--skip-filter`: Skip dataset filtering step (assumes datasets already filtered)
- `--skip-training`: Skip training entirely (for testing pipeline)
- `--skip-existing`: Skip training models that already have checkpoints
- `--device N`: GPU device to use (default: 0)
- `--patience N`: Early stopping patience in epochs (default: 10)

### Manual Step-by-Step Execution

If you prefer to run steps manually:

#### Step 1: Filter the Precomputed Datasets

```bash
# Run the filtering script (will overwrite datasets in place)
python filter_features.py
```

This will process datasets in `data/datasets_extra/`:
- `transformer/train_dataset.pkl` ✓
- `transformer/val_dataset.pkl` ✓
- `transformer/test_dataset.pkl` ✓
- `zoo/train_dataset.pkl` (no changes - zoo doesn't use game state)
- `zoo/val_dataset.pkl` (no changes)
- `zoo/test_dataset.pkl` (no changes)

#### Step 2: Train the Model

```bash
# Train transformer model
python src/train.py --model_type transformer --device 0 --patience 10
```

**Training script flags:**
- `--model_type`: "transformer" or "zoo" (use "transformer")
- `--device N`: GPU device (0, 1, etc.) or -1 for CPU
- `--patience N`: Early stopping patience (default: 10)
- `--skip-existing`: Skip training models with existing checkpoints
- `--shuffle` or `-S`: Shuffle hyperparameter search order
- `--reverse` or `-R`: Reverse hyperparameter search order
- `--hparam_search_iters N`: Limit hyperparameter search to N combinations

### Example Commands

```bash
# Full hyperparameter search (12 configurations)
python src/train.py --model_type transformer --device 0 --patience 10

# Skip existing checkpoints
python src/train.py --model_type transformer --device 0 --skip-existing

# Run on CPU
python src/train.py --model_type transformer --device -1

# Limited hyperparameter search (3 random configs)
python src/train.py --model_type transformer --device 0 --hparam_search_iters 3

# Shuffle search order for parallel runs
python src/train.py --model_type transformer --device 0 --shuffle
```

## Hyperparameter Search Space

The training script will search over:
- **Learning rates**: [1e-4]
- **Model dimensions**: [32, 128, 512]
- **Number of layers**: [1, 2, 4, 8]
- **Total configurations**: 12 per architecture

## Output

Trained models and checkpoints will be saved to:
- Local: `models/transformer/M{model_dim}_L{num_layers}_LR{lr}/checkpoints/`
- Google Drive (if available): `/content/drive/MyDrive/SportsTrackingTransformer/models/`

Results will be saved as:
- `epoch={N}-val_loss={X.XXX}.ckpt`
- `epoch={N}-val_loss={X.XXX}.results.parquet`

View training progress:
```bash
tensorboard --logdir models/transformer
```

## Feature Indices Reference

Original 11 features (from add-game-state-features):
```
[0] x_rel
[1] y_rel
[2] vx
[3] vy
[4] side
[5] is_ball_carrier
[6] yardsToGo          ❌ REMOVED
[7] down               ❌ REMOVED
[8] distanceToGoal     ✅ KEPT
[9] quarter            ❌ REMOVED
[10] half_seconds_remaining ❌ REMOVED
```

New 7 features (23-entity branch):
```
[0] x_rel
[1] y_rel
[2] vx
[3] vy
[4] side
[5] is_ball_carrier
[6] distanceToGoal
```

## Troubleshooting

### If datasets are missing:
Check if the datasets exist in `data/datasets_extra/`:
```bash
ls -lh data/datasets_extra/transformer/
ls -lh data/datasets_extra/zoo/
```

### If you need to regenerate from scratch:
```bash
# 1. Switch to add-game-state-features branch
git checkout add-game-state-features

# 2. Prepare raw data
python src/prep_extra_data.py

# 3. Create datasets (11 features)
python src/datasets.py

# 4. Switch back to 23-entity branch
git checkout 23-entity

# 5. Filter datasets (11 → 7 features)
python filter_features.py

# 6. Train
./run_transformer_only.py --skip-filter
```

### Verify feature dimensions:
You can quickly test if the datasets have the correct shape:
```python
import pickle
from pathlib import Path

with open("data/datasets_extra/transformer/test_dataset.pkl", "rb") as f:
    ds = pickle.load(f)

# Check a sample feature array
key = ds.keys[0]
print(f"Feature shape: {ds.feature_arrays[key].shape}")
# Should print: Feature shape: (22, 7)
```

### Common Issues

**Import error when running filter_features.py:**
```bash
# Make sure you're in the project root and datasets.py is importable
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python filter_features.py
```

**Out of memory during training:**
```bash
# Reduce batch size in src/train.py (line 267)
# Or limit hyperparameter search
python src/train.py --model_type transformer --device 0 --hparam_search_iters 3
```

**Checkpoints not saving:**
```bash
# Check if models directory exists and has write permissions
mkdir -p models/transformer
ls -ld models/
```
