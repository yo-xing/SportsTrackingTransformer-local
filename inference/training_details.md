# Training Details: 23-Entity Football Model

## Overview

This document provides technical details about the training pipeline used to create the model checkpoint included in this inference folder.

**Branch**: `23-entity-football`
**Model**: M64_L4_LR1e-04
**Checkpoint**: epoch=41-val_loss=2.730.ckpt

For complete repository documentation, see the main [README](../README.md).

---

## Model Architecture

### Sports Transformer

The model uses a custom Transformer architecture designed for sports tracking data:

- **Input**: 23 entities per frame
  - 22 players (11 offense + 11 defense)
  - 1 football

- **Features**: 7 features per entity
  - **Entity features (6)**: `x_rel`, `y_rel`, `vx`, `vy`, `side`, `is_ball_carrier`
  - **Game state feature (1)**: `distanceToGoal`

- **Output**: 110-class probability distribution
  - Classes represent yards gained from -10 to +99
  - Model predicts at every frame as the play unfolds

### Model Variants

The training pipeline supports multiple model configurations:

| Dimension (M) | Layers (L) | Learning Rate | Parameters |
|---------------|------------|---------------|------------|
| M32 | L1, L2, L4, L8 | 1e-04, 1e-03 | ~50K-200K |
| M64 | L1, L2, L4, L8 | 1e-04, 1e-03 | ~200K-800K |
| M128 | L1, L2, L4, L8 | 1e-04, 1e-03 | ~800K-3M |
| M512 | L1, L2, L4, L8 | 1e-04, 1e-03 | ~13M-50M |

**Selected Configuration**: M64_L4_LR1e-04
- Model dimension: 64
- Number of layers: 4
- Learning rate: 1e-04
- Total parameters: ~800K

---

## Feature Descriptions

### Entity Features (6)

1. **x_rel**: Player x-position relative to ball carrier (normalized)
   - Allows model to learn position-based patterns independent of field location

2. **y_rel**: Player y-position relative to ball carrier (normalized)
   - Complements x_rel for full 2D spatial awareness

3. **vx**: Player velocity in x-direction (yards/sec)
   - Cartesian format for direct learning of movement patterns

4. **vy**: Player velocity in y-direction (yards/sec)
   - Captures perpendicular movement (e.g., lateral cuts)

5. **side**: Team indicator
   - Values: `1` (offense), `-1` (defense), `0` (football)

6. **is_ball_carrier**: Binary flag indicating ball carrier
   - Only one player per play has value `1`, all others `0`

### Game State Feature (1)

7. **distanceToGoal**: Yards from goal line
   - Provides field position context
   - Helps model understand field compression near end zones

---

## Data Pipeline

The complete data transformation pipeline from raw data to trained model:

### Step 1: Raw Data Collection

**Source**: Axially format parquet files
**Location**: `extra_data/Week XX/*.parquet`

Raw data schema includes:
- Player tracking: `gamekey`, `playid`, `nfl_id`, `frame_id`, `x`, `y`, `vel_x`, `vel_y`
- Game context: `play_type`, `event`, `yards_gained`, `line_of_scrimmage`
- Player info: `display_name`, `team_abbr`, `roster_position`, `possession_status`

See [README.md](README.md#understanding-the-sample-data) for complete schema.

### Step 2: Data Preparation (`src/prep_extra_data.py`)

Transforms raw Axially format to BDB 2024 format:

1. **Column Mapping**: Snake_case → camelCase
   - `gamekey` → `gameId`, `playid` → `playId`, etc.

2. **Filtering**:
   - Remove football rows (`possession_status='ball'`)
   - Keep only pass plays (`play_type='play_type_pass'`)

3. **Ball Carrier Identification**:
   - Find offensive player closest to football at handoff/snap
   - Fallback to first_contact or ball_snap events

4. **Feature Engineering**:
   - Calculate relative positions (x_rel, y_rel)
   - Add derived features
   - Normalize coordinates and velocities

5. **Play Direction Standardization**:
   - Orient all plays left-to-right for consistency
   - Mirror plays attacking right endzone

6. **Data Augmentation**:
   - Mirror all plays horizontally
   - Doubles dataset size
   - Improves left/right symmetry learning

**Output**: `data/split_prepped_data_extra/*.parquet`

### Step 3: Feature Filtering (`filter_features.py`)

Reduces from 11 features → 7 features:

**Removed features** (from add-game-state-features branch):
- `yardsToGo`
- `down`
- `quarter`
- `half_seconds_remaining`

**Kept features**:
- 6 entity features (x_rel, y_rel, vx, vy, side, is_ball_carrier)
- 1 game state feature (distanceToGoal)

**Rationale**: Minimize feature complexity while retaining essential field position context.

### Step 4: Dataset Precomputation (`src/datasets.py`)

Converts parquet files to PyTorch-ready tensors:

1. **Load Data**: Read features and targets from parquet
2. **Tensor Conversion**: Convert to PyTorch tensors
3. **Caching**: Save as pickle files for fast loading
4. **Train/Val/Test Split**: 70% / 15% / 15%

**Output**: `data/datasets_extra_norm_football/transformer/*_dataset.pkl`

### Step 5: Training (`src/train.py`)

PyTorch Lightning training loop:

**Framework**: PyTorch Lightning 2.4.0
**Device**: GPU (CUDA)
**Loss Function**: CrossEntropyLoss
**Optimizer**: AdamW
**Batch Size**: 1024
**Max Epochs**: 100

**Training Features**:
- Early stopping (patience=10 epochs)
- Model checkpointing (saves best model by validation loss)
- TensorBoard logging
- Learning rate scheduling

**Training Time**: ~2-4 hours per configuration on GPU

### Step 6: Results Generation (`src/generate_results_summary.py`)

Post-training analysis:

1. **Inference**: Run predictions on train/val/test splits
2. **Metrics Calculation**: MAE, RMSE, ADE
3. **Visualizations**: Prediction accuracy plots
4. **CSV Export**: Detailed results for analysis

### Step 7: Model Selection (`src/pick_best_models.py`)

Selects best checkpoint per configuration based on validation loss.

---

## Training Configuration

### Hardware & Software

- **GPU**: NVIDIA CUDA-capable GPU (required)
- **Python**: 3.12+
- **PyTorch**: 2.3.0
- **PyTorch Lightning**: 2.4.0
- **Data Processing**: Polars 1.5.0

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-04 | Lower rate for stable convergence |
| Batch Size | 1024 | Balanced for GPU memory and training speed |
| Early Stopping Patience | 10 epochs | Prevents overfitting |
| Max Epochs | 100 | Typically stops around epoch 40-50 |
| Optimizer | AdamW | Adam with weight decay |
| Loss Function | CrossEntropyLoss | Standard for classification |

### Data Split

- **Training Set**: 70% of plays (~40K plays, ~1.6M frames)
- **Validation Set**: 15% of plays (~8.5K plays, ~340K frames)
- **Test Set**: 15% of plays (~8.5K plays, ~340K frames)

Split at **play level** (not frame level) to prevent data leakage.

---

## Model Performance

### Selected Model: M64_L4_LR1e-04

**Validation Loss**: 2.730 (epoch 41)

**Why This Model**:
- Balanced complexity (64 dimensions, 4 layers)
- Good generalization (low validation loss)
- Efficient inference (~800K parameters)
- Stable training (converged at epoch 41)

### Evaluation Metrics

**MAE (Mean Absolute Error)**:
- Measures average yards prediction error
- More interpretable than cross-entropy
- Example: MAE=3.5 means "off by 3.5 yards on average"

**Cross-Entropy Loss**:
- Measures quality of probability distribution
- Used for training and model selection
- Lower is better

### Performance by Event Type

Model predictions improve as plays progress:
- **Early frames** (ball_snap): Higher uncertainty
- **Middle frames** (handoff, pass_forward): Moderate accuracy
- **Late frames** (tackle, touchdown): Best accuracy

### Comparison to Baseline

The 23-entity-football model includes football tracking as the 23rd entity, compared to baseline models with only 22 players. This provides:
- Explicit football position information
- Better understanding of pass trajectories
- Improved accuracy for incomplete passes

---

## File Locations Referenced

| Component | File Path | Lines |
|-----------|-----------|-------|
| Training Script | `src/train.py` | 1-400 |
| Model Architecture | `src/models.py` | 276-407 |
| Data Preparation | `src/prep_extra_data.py` | 1-572 |
| Dataset Loading | `src/datasets.py` | 59-195 |
| Feature Filtering | `filter_features.py` | 1-150 |
| Pipeline Orchestration | `run_transformer_only.py` | 1-368 |

**Branch**: 23-entity-football

---

## Differences from Main Branch

### Main Branch (22 entities, 6 features)
- **Entities**: 22 players only
- **Features**: 6 (x_rel, y_rel, vx, vy, side, is_ball_carrier)
- **Game State**: None
- **Design**: Minimal feature engineering

### 23-Entity-Football Branch (23 entities, 7 features)
- **Entities**: 22 players + 1 football
- **Features**: 7 (6 entity + distanceToGoal)
- **Game State**: Distance to goal
- **Design**: Football tracking + minimal game context

**Why Track Football**:
- Provides explicit position information
- Captures pass trajectories
- Improves incomplete pass predictions
- Minimal overhead (1 additional entity)

---

## Reproducing Training

**Note**: This section is for reference only. The inference folder does NOT require retraining.

### Prerequisites

1. Python 3.12+
2. NVIDIA GPU with CUDA support
3. 16GB+ RAM
4. ~50GB disk space for data and checkpoints

### Installation

```bash
# Clone repository
git clone <repository-url>
cd SportsTrackingTransformer

# Checkout branch
git checkout 23-entity-football

# Install dependencies
pip install -r requirements.txt
```

### Running Full Pipeline

```bash
# Run complete pipeline (preprocessing + training + results)
python run_transformer_only.py --device 0
```

**Duration**: ~6-8 hours for all model configurations

### Training Single Configuration

```bash
# Train specific model
python src/train.py \
  --model_type transformer \
  --model_dim 64 \
  --num_layers 4 \
  --lr 1e-4 \
  --batch_size 1024 \
  --device cuda:0
```

### Google Drive Paths (Colab Environment)

If running on Google Colab, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Paths will resolve to `/content/drive/MyDrive/...`

**Local Environment**: Modify paths in source files to use local directories.

---

## Additional Resources

- **Main README**: [../README.md](../README.md)
- **Inference Guide**: [README.md](README.md)
- **Visualization Notebook**: [visualize.ipynb](visualize.ipynb)
- **Play Visualization**: [visualize_play.py](visualize_play.py)

---

**Questions or Issues?**
File an issue in the repository or consult the main documentation.
