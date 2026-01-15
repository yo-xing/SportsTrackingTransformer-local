# Cache Locations for 23-Entity Branch

## Overview

This document explains the cache directory structure to ensure proper separation between branches and avoid conflicts.

## Directory Structure

### add-game-state-features Branch (Source - 11 features)
This branch creates the source datasets with 11 features (23 entities):

**Local directories:**
- Prepped data: `data/split_prepped_data_extra_gamestate_23/`
- Datasets (pickled): `data/datasets_extra_gamestate_23/`

**Google Drive cache:**
- Prepped data: `/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_gamestate_23/`
- Datasets: `/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_gamestate_23/`

**Features:** 11 total
- x_rel, y_rel, vx, vy, side, is_ball_carrier (6 player features)
- yardsToGo, down, distanceToGoal, quarter, half_seconds_remaining (5 game state features)

**Entities:** 23 (22 players + 1 football)

---

### 23-entity Branch (Target - 7 features)
This branch uses filtered datasets with only 7 features (22 entities):

**Local directories:**
- Prepped data: `data/split_prepped_data_extra/` (NOT USED - we filter pickled datasets)
- Datasets (pickled): `data/datasets_extra/`

**Google Drive cache:**
- Datasets: `/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_7feat/`
- Models: `/content/drive/MyDrive/SportsTrackingTransformer/models/`

**Features:** 7 total
- x_rel, y_rel, vx, vy, side, is_ball_carrier (6 player features)
- distanceToGoal (1 game state feature)

**Entities:** 22 (22 players, no football)

---

## Data Flow

```
┌─────────────────────────────────────┐
│ add-game-state-features Branch      │
│                                     │
│ datasets_extra_gamestate_23/        │
│ ├── transformer/                    │
│ │   ├── train_dataset.pkl (11 feat)│
│ │   ├── val_dataset.pkl   (11 feat)│
│ │   └── test_dataset.pkl  (11 feat)│
│ └── zoo/                            │
│     └── ...                         │
└─────────────────────────────────────┘
              │
              │ filter_features.py
              │ (keeps indices 0-5, 8)
              ▼
┌─────────────────────────────────────┐
│ 23-entity Branch                    │
│                                     │
│ datasets_extra/                     │
│ ├── transformer/                    │
│ │   ├── train_dataset.pkl (7 feat) │
│ │   ├── val_dataset.pkl   (7 feat) │
│ │   └── test_dataset.pkl  (7 feat) │
│ └── zoo/                            │
│     └── ...                         │
└─────────────────────────────────────┘
              │
              │ train.py
              ▼
┌─────────────────────────────────────┐
│ Models Directory                    │
│                                     │
│ models/transformer/                 │
│ └── M{dim}_L{layers}_LR{lr}/        │
│     └── checkpoints/                │
│         ├── epoch=X-val_loss=Y.ckpt │
│         └── *.results.parquet       │
└─────────────────────────────────────┘
```

## File Configuration

### filter_features.py
```python
INPUT_DIR = Path("data/datasets_extra_gamestate_23/")  # From add-game-state-features
OUTPUT_DIR = Path("data/datasets_extra/")              # For 23-entity
```

### src/datasets.py (23-entity branch)
```python
PREPPED_DATA_DIR = Path("data/split_prepped_data_extra/")
DATASET_DIR = Path("data/datasets_extra/")
DRIVE_DIR = Path("/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_7feat")
```

### src/datasets.py (add-game-state-features branch)
```python
PREPPED_DATA_DIR = Path("data/split_prepped_data_extra_gamestate_23/")
DATASET_DIR = Path("data/datasets_extra_gamestate_23/")
DRIVE_DIR = Path("/content/drive/MyDrive/NewDataSportsTrackingTransformer_cache_gamestate_23")
```

## Workflow

### Initial Setup (One Time)

1. **On add-game-state-features branch** - Create 11-feature datasets:
   ```bash
   git checkout add-game-state-features
   # If datasets don't exist, run:
   python src/prep_extra_data.py
   python src/datasets.py
   ```

2. **Switch to 23-entity branch** - Filter to 7 features:
   ```bash
   git checkout 23-entity
   python filter_features.py
   ```

### Training (23-entity branch)

```bash
# Run the complete pipeline
./run_transformer_only.py

# Or manually:
python src/train.py --model_type transformer --device 0
```

## Verification Commands

### Check if source datasets exist (11 features):
```bash
ls -lh data/datasets_extra_gamestate_23/transformer/
# Should show: train_dataset.pkl, val_dataset.pkl, test_dataset.pkl
```

### Check if filtered datasets exist (7 features):
```bash
ls -lh data/datasets_extra/transformer/
# Should show: train_dataset.pkl, val_dataset.pkl, test_dataset.pkl
```

### Verify feature dimensions:
```python
import pickle

# Check source (11 features)
with open("data/datasets_extra_gamestate_23/transformer/test_dataset.pkl", "rb") as f:
    ds = pickle.load(f)
key = ds.keys[0]
print(f"Source shape: {ds.feature_arrays[key].shape}")
# Should print: Source shape: (23, 11) or (22, 11)

# Check filtered (7 features)
with open("data/datasets_extra/transformer/test_dataset.pkl", "rb") as f:
    ds = pickle.load(f)
key = ds.keys[0]
print(f"Filtered shape: {ds.feature_arrays[key].shape}")
# Should print: Filtered shape: (22, 7)
```

## Common Issues

### Issue: "datasets_extra_gamestate_23 not found"
**Solution:** You need to create the source datasets first from add-game-state-features branch:
```bash
git checkout add-game-state-features
python src/datasets.py
git checkout 23-entity
python filter_features.py
```

### Issue: "Feature shape mismatch"
**Solution:** Make sure you're using the correct branch:
- add-game-state-features: expects (23, 11) or (22, 11) arrays
- 23-entity: expects (22, 7) arrays

### Issue: "Overwriting wrong datasets"
**Solution:** Always verify INPUT_DIR and OUTPUT_DIR in filter_features.py:
- INPUT_DIR should be: `data/datasets_extra_gamestate_23/`
- OUTPUT_DIR should be: `data/datasets_extra/`

## Disk Space

Approximate sizes:
- Source datasets (11 features): ~500MB per split = ~1.5GB total
- Filtered datasets (7 features): ~320MB per split = ~960MB total
- Total: ~2.5GB for both branches' datasets

If disk space is limited, you can delete the source datasets after filtering:
```bash
# ONLY do this after verifying filtered datasets work!
rm -rf data/datasets_extra_gamestate_23/
```
