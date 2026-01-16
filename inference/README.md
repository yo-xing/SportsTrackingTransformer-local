# Sports Tracking Transformer - Inference Folder

Simple, self-contained inference setup for the 23-entity football model (Min+Football).

**Model**: M64_L4_LR1e-04 | **Test MAE**: 4.44 yards | **Checkpoint**: epoch=41-val_loss=2.730.ckpt

---

## Quick Start

Run inference in 3 commands:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add model checkpoint (see Model Setup below)

# 3. Run inference
python run_inference.py
```

**Output**: `predictions.csv` with predicted yards for each frame

---

## What's in This Folder

```
inference/
├── README.md                    # This file - setup and usage guide
├── MODEL_WRITEUP.html          # Complete model analysis and results
├── training_details.md         # Technical training pipeline details
├── requirements.txt            # Python dependencies
├── run_inference.py           # Main inference script
├── visualize.ipynb            # Jupyter notebook for analyzing predictions
├── visualize_play.py          # Script for play-by-play visualization
├── sample_data/               # Sample tracking data
│   └── Week 01/58503.parquet  # Raw Axially format data
├── model/                     # Model checkpoint (you provide)
│   └── [place .ckpt here]
├── visualizations/            # Output folder for plots
└── predictions.csv            # Generated predictions (after running)
```

---

## Model Setup

### Step 1: Get Model Checkpoint

**You need to download the model checkpoint from Google Drive:**

**Location**: `/content/drive/MyDrive/SportsTrackingTransformer/models_norm_football/transformer/M64_L4_LR1e-04/checkpoints/`

**File**: `epoch=41-val_loss=2.730.ckpt` (2.49 MB)

**Place in**: `inference/model/`

```bash
# After downloading from Google Drive:
cp ~/Downloads/epoch=41-val_loss=2.730.ckpt inference/model/
```

### Step 2: Verify Setup

```bash
# Check that everything is in place:
ls inference/model/          # Should show: epoch=41-val_loss=2.730.ckpt
ls inference/sample_data/    # Should show: Week 01/
```

---

## Prerequisites

- **Python**: 3.12+
- **Hardware**: CPU (slower) or GPU (faster)
- **Disk Space**: ~50 MB for dependencies + model
- **No Cloud Required**: Runs completely locally

### Installation

```bash
# Option 1: Using pip
pip install -r requirements.txt

# Option 2: Using uv (faster)
uv pip install -r requirements.txt

# Option 3: Create virtual environment first (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running Inference

### Basic Usage

```bash
python run_inference.py
```

**Expected Runtime**:
- CPU: 2-5 minutes
- GPU: 30-60 seconds

**Output**:
```
==============================================================
Sports Tracking Transformer - Simple Inference
==============================================================

[1/4] Checking sample data...
Found raw Axially-format data:
  58503.parquet (3.9 MB)

[2/4] Loading model...
Loading model from: epoch=41-val_loss=2.730.ckpt
Using GPU: NVIDIA GeForce RTX 3090
Model Configuration:
  Architecture: transformer
  Model Dimension: 64
  Layers: 4
  Learning Rate: 0.0001
  Total Parameters: 203,520
  Trainable Parameters: 203,520

[3/4] Running inference on train set...
  Loading from sample files...
  Created dataset with 8453 frames
  Running inference on 8453 samples...

  TRAIN Metrics:
    MAE: 4.32 yards
    Mean predicted: 5.87 yards
    Mean actual: 6.01 yards

[4/4] Saving predictions...
  Saved to: inference/predictions.csv

==============================================================
Inference Complete!
==============================================================
Total predictions: 25,359
  train: 8,453 predictions, MAE: 4.32 yards
  val: 8,453 predictions, MAE: 4.34 yards
  test: 8,453 predictions, MAE: 4.44 yards

Results saved to: inference/predictions.csv
```

### Understanding the Output

**predictions.csv** contains:

| Column | Description | Example |
|--------|-------------|---------|
| `dataset_split` | Data split (train/val/test) | train |
| `gameId` | Unique game identifier | 58503 |
| `playId` | Play number within game | 75 |
| `frameId` | Frame number (10 fps) | 23 |
| `predicted_yards` | Expected yards (probabilistic) | 6.42 |
| `predicted_class_yards` | Most likely yards (class) | 7 |
| `confidence` | Confidence in prediction | 0.23 |
| `actual_yards` | Ground truth yards gained | 7 |

---

## Visualizing Results

### Jupyter Notebook (Recommended)

```bash
jupyter notebook visualize.ipynb
```

**What you'll see**:
1. Prediction accuracy metrics (MAE, RMSE)
2. Predicted vs actual scatter plots
3. Error distribution analysis
4. Confidence calibration
5. Sample play predictions

All plots automatically saved to `visualizations/` folder.

### Play-by-Play Visualization

```bash
python visualize_play.py
```

**Note**: This script is designed for Google Colab and requires Google Drive paths. See the script header for setup instructions.

---

## Understanding the Sample Data

### Raw Data Schema (Axially Format)

The sample data (`sample_data/Week 01/58503.parquet`) is in Axially format:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `gamekey` | Int | Unique game identifier | 58503 |
| `playid` | Int | Play number within game | 75 |
| `nfl_id` | Int | NFL player ID | 2495116 |
| `frame_id` | Int | Frame number (10 fps) | 1 |
| `x` | Float | Field x-coordinate (0-120 yards) | 55.3 |
| `y` | Float | Field y-coordinate (0-53.3 yards) | 26.7 |
| `vel_x` | Float | X velocity (yards/sec) | 2.1 |
| `vel_y` | Float | Y velocity (yards/sec) | -0.5 |
| `vel` | Float | Speed magnitude (yards/sec) | 2.16 |
| `o` | Float | Orientation angle (degrees) | 90.5 |
| `possession_status` | String | 'off', 'def', 'ball' | off |
| `play_type` | String | Type of play | play_type_pass |
| `event` | String | Event at frame | ball_snap |
| `yards_gained` | Float | Yards gained on play | 7.5 |
| `line_of_scrimmage` | Float | LOS position (yards) | 35.0 |
| `display_name` | String | Player name | J.Smith |
| `team_abbr` | String | Team abbreviation | SF |
| `roster_position` | String | Position | QB |

**Data Format**:
- One row per player per frame
- 22 players × ~40 frames/play ≈ 880 rows per play
- Football rows have `possession_status='ball'` (filtered during preprocessing)

**Coordinate System**:
- **X-axis**: 0-120 yards (length of field)
- **Y-axis**: 0-53.3 yards (width of field)
- **Frame Rate**: 10 frames per second
- **Velocity**: Cartesian format (vx, vy) in yards/sec

---

## Using Your Own Data

### Option A: Match Axially Format (Easiest)

1. **Prepare your data** in the same format as the sample (see schema above)
2. **Place files** in `inference/sample_data/Week 01/`
3. **Run inference**: `python run_inference.py`

The script will automatically preprocess your data.

**Critical Requirements**:
- ✅ All columns from schema table must be present
- ✅ At least 22 players per frame (11 offense + 11 defense)
- ✅ Ball carrier identifiable (player close to football at snap/handoff)
- ✅ Velocity in Cartesian format (vx, vy), not polar
- ✅ Yards gained available for target labels

### Option B: Use Preprocessed Data

If you already have data preprocessed in BDB 2024 format:

1. **Create sample files** with required columns:
   - `gameId`, `playId`, `frameId`, `nflId`
   - `x_rel`, `y_rel`, `vx`, `vy`
   - `side`, `is_ball_carrier`, `distanceToGoal`
   - `mirrored` (for filtering)

2. **Split into features and targets**:
   - `inference/sample_data/train_features_sample.parquet`
   - `inference/sample_data/train_targets_sample.parquet`
   - (same for val and test)

3. **Run inference**: `python run_inference.py`

### Data Validation

The script validates:
- ✅ Column names and types
- ✅ No missing critical values
- ✅ Player counts per frame (warns if not 22)
- ✅ Yards gained in valid range (-10 to +99)

---

## Model Details

### Architecture: Min+Football

**Why This Model?**
- **Best Performance**: 4.44 yards test MAE (10.7% better than baseline)
- **Football Tracking**: 23rd entity provides 2% improvement over 22-player model
- **Minimal Features**: 7 features vs 11 - simpler is better
- **Strong Generalization**: Train 4.32y → Val 4.34y → Test 4.44y

See [MODEL_WRITEUP.html](MODEL_WRITEUP.html) for complete analysis.

### Features (7 per entity)

**Entity Features (6)**:
1. `x_rel`: Position relative to ball carrier (x-axis)
2. `y_rel`: Position relative to ball carrier (y-axis)
3. `vx`: Velocity in x-direction (yards/sec)
4. `vy`: Velocity in y-direction (yards/sec)
5. `side`: Team (1=offense, -1=defense, 0=football)
6. `is_ball_carrier`: Binary flag (1=carrier, 0=other)

**Game State Feature (1)**:
7. `distanceToGoal`: Yards from goal line

### Entities Tracked (23)

- **22 Players**: 11 offense + 11 defense
- **1 Football**: Tracked as 23rd entity with same features

### Output (110 Classes)

- **Classes**: Yards gained from -10 to +99
- **Prediction**: Probability distribution over all classes
- **Expected Yards**: Weighted average of distribution
- **Confidence**: Maximum probability value

### Model Specifications

| Specification | Value |
|---------------|-------|
| Architecture | Transformer |
| Model Dimension | 64 |
| Layers | 4 |
| Parameters | ~200K |
| Training Epochs | 41 |
| Validation Loss | 2.730 |
| Test MAE | 4.44 yards |

---

## Performance Metrics

### Overall Accuracy

| Split | MAE | Samples |
|-------|-----|---------|
| Train | 4.32 yards | 1,994,318 |
| Val | 4.34 yards | 429,224 |
| Test | 4.44 yards | 433,398 |

**Minimal Overfitting**: Train → Test gap of only 0.12 yards

### By Play Type

| Play Type | MAE | Notes |
|-----------|-----|-------|
| Rush | 2.97 yards | Very accurate (within 9 feet) |
| Pass | 5.85 yards | More uncertainty (routes, QB) |
| Sack | 4.12 yards | Moderate accuracy |

### By Frame Timing

| Frame Position | MAE | Notes |
|----------------|-----|-------|
| First 10 frames | 6.23 yards | High uncertainty early |
| Middle frames | 5.01 yards | Moderate accuracy |
| Last 10 frames | 1.58 yards | Pinpoint accuracy (5 feet) |

**Key Insight**: Model accuracy improves as play unfolds.

---

## Troubleshooting

### GPU Not Available

**Symptom**: Message "No GPU detected, using CPU"

**Solution**: This is expected if you don't have NVIDIA GPU with CUDA. The model will run on CPU (slower but works fine).

### Out of Memory

**Symptom**: RuntimeError about CUDA out of memory

**Solutions**:
1. Reduce batch size in `run_inference.py`: Change `batch_size=256` to `batch_size=128`
2. Use CPU instead: Set device to CPU in model loading

### Missing Dependencies

**Symptom**: `ModuleNotFoundError: No module named 'polars'`

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Format Errors

**Symptom**: KeyError or column not found errors

**Solution**:
1. Check your data matches the Axially format schema
2. Ensure all required columns are present
3. Verify column names are exact (case-sensitive)

### Model Not Found

**Symptom**: `FileNotFoundError: No .ckpt files found`

**Solution**: Download and place model checkpoint in `inference/model/`:
```bash
cp ~/Downloads/epoch=41-val_loss=2.730.ckpt inference/model/
```

---

## Advanced Usage

### Batch Processing Multiple Games

To process multiple game files:

```python
# Place multiple parquet files in sample_data/Week 01/
# The script will automatically process all files
```

### Modifying Batch Size

For faster inference on GPU or memory constraints:

```python
# Edit run_inference.py, line ~450:
dataloader = DataLoader(
    dataset,
    batch_size=128,  # Change from 256 to 128
    ...
)
```

### Filtering by Play Type

To analyze only specific play types:

```python
# After inference, filter predictions.csv:
import pandas as pd

df = pd.read_csv('predictions.csv')
# Filter by play type (requires joining with original data)
```

---

## Additional Resources

### Documentation

- **[MODEL_WRITEUP.html](MODEL_WRITEUP.html)** - Complete model analysis and comparison
- **[training_details.md](training_details.md)** - Technical training pipeline details
- **[visualize.ipynb](visualize.ipynb)** - Interactive visualization notebook
- **[visualize_play.py](visualize_play.py)** - Play-by-play visualization script

### Repository

- **Main README**: [../README.md](../README.md)
- **Source Code**: [../src/](../src/)
- **Training Scripts**: [../src/train.py](../src/train.py)

### Model Comparison

This inference folder uses the **Min+Football** model (23-entity-norm). See MODEL_WRITEUP.html for comparison with:
- Baseline (6 features, 22 players)
- Minimalist (7 features, 22 players, no football)
- Contextual (11 features, 23 entities)

---

## Citation & Acknowledgments

If you use this model in your research, please cite:

```
Sports Tracking Transformer (Min+Football)
M64_L4_LR1e-04, epoch=41
Training Data: 18 weeks NFL tracking data
Test MAE: 4.44 yards
```

**Data Source**: NFL RFID tracking data (Axially format)
**Framework**: PyTorch Lightning 2.4.0
**Architecture**: Transformer with self-attention

---

## Support

**Questions or Issues?**
1. Check [troubleshooting section](#troubleshooting) above
2. Review [MODEL_WRITEUP.html](MODEL_WRITEUP.html) for model details
3. See [training_details.md](training_details.md) for technical specs
4. File an issue in the repository

---

**Last Updated**: January 2026
**Model Version**: 23-entity-football (Min+Football)
**Checkpoint**: epoch=41-val_loss=2.730.ckpt
