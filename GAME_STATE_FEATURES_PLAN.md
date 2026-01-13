# Game State Features Implementation Plan

## Overview
Add game state features (yardsToGo, down, distanceToGoal, quarter, half_seconds_remaining) to the transformer model to provide situational context about each play.

## Current State
**Current model features (6 per player):**
- `x_rel`: Relative x position from ball
- `y_rel`: Relative y position from ball
- `vx`: Velocity in x direction
- `vy`: Velocity in y direction
- `side`: 1 for offense, -1 for defense
- `is_ball_carrier`: 1 if ball carrier, 0 otherwise

**Input shape:** `[batch_size, 22 players, 6 features]`

## Proposed Features
**Game state features (constant across all 22 players):**
- `yardsToGo`: Distance for first down (1-99, typically 1-20)
- `down`: Current down (1-4)
- `distanceToGoal`: Distance to opponent's goal (0-100)
- `quarter`: Quarter (1-4)
- `half_seconds_remaining`: Seconds remaining in current half (0-1800)

Most of these are already calculated in `prep_extra_data.py` and saved to the features parquet files.

## Implementation Options

### Option 1: Broadcast to All Players (Recommended)
Append game state to each player's feature vector.

**New input shape:** `[batch_size, 22 players, 11 features]`

**Pros:**
- Simple implementation
- Transformer naturally learns relationships between game state and player positions
- No architectural changes needed beyond input dimension

**Cons:**
- Redundant data (same 5 values repeated 22 times)
- Slightly larger memory footprint

**Implementation:**
1. Modify `datasets.py` to include game state columns in feature extraction
2. Update `models.py` to accept `feature_len=11` for transformer
3. Retrain models

### Option 2: Separate Game State Embedding
Create a separate embedding for game state and concatenate with pooled player features.

**Architecture:**
```
Input: [B, 22, 6] player features + [B, 5] game state
↓
Player branch: [B, 22, 6] → Transformer → Pool → [B, model_dim]
Game state branch: [B, 5] → Linear → [B, model_dim]
↓
Concatenate: [B, 2*model_dim]
↓
Decoder: [B, 2*model_dim] → [B, NUM_YARDS_CLASSES]
```

**Pros:**
- More parameter efficient
- Clearer separation of concerns
- Could use different learning rates for each branch

**Cons:**
- More complex architecture
- Game state processed independently of player interactions
- May not capture game-state-dependent player behaviors as well

### Option 3: Add Game State to Pooled Features
Process players through transformer, then concatenate game state before decoder.

**Architecture:**
```
Input: [B, 22, 6] player features + [B, 5] game state
↓
Player features: [B, 22, 6] → Transformer → Pool → [B, model_dim]
↓
Concatenate with game state: [B, model_dim + 5]
↓
Decoder: [B, model_dim + 5] → [B, NUM_YARDS_CLASSES]
```

**Pros:**
- Game state informs final prediction
- Simpler than Option 2

**Cons:**
- Game state doesn't influence transformer attention
- May miss game-state-dependent player relationships

## Recommended Approach: Option 1

**Reasoning:**
- Transformers excel at learning relationships in sequence data
- Broadcasting allows the model to learn how game state affects each player's importance
- Example: On 3rd-and-1, the model can learn to weight offensive linemen differently than on 1st-and-10
- Minimal code changes
- Let the model figure out what's important through attention

## Implementation Steps

### 1. Update Data Preparation (datasets.py)
Currently, `datasets.py` extracts only specific columns. Need to include game state.

**File:** `src/datasets.py`
**Location:** Feature extraction in `SportsDataset.__init__()` or similar

**Changes needed:**
```python
# Current: Extracts 6 features
TRANSFORMER_FEATURES = ['x_rel', 'y_rel', 'vx', 'vy', 'side', 'is_ball_carrier']

# New: Add game state features
TRANSFORMER_FEATURES = [
    'x_rel', 'y_rel', 'vx', 'vy', 'side', 'is_ball_carrier',
    'yardsToGo', 'down', 'distanceToGoal', 'quarter', 'half_seconds_remaining'
]
```

### 2. Update Model Architecture (models.py)
**File:** `src/models.py`
**Location:** `LitModel.__init__()` line 306

**Changes needed:**
```python
# Current
self.feature_len = 6 if self.model_type == "transformer" else 10

# New
self.feature_len = 11 if self.model_type == "transformer" else 10  # Transformer now uses 11
```

**Note:** The SportsTransformer model itself doesn't need changes - it already accepts `feature_len` as a parameter and will automatically adjust to the new input dimension.

### 3. Verify Feature Availability
Check that all game state features are present in the prepared data.

**Columns to verify in features parquet:**
- `yardsToGo` - Already mapped from `yards_to_go` in prep_extra_data.py
- `down` - Need to check if this exists
- `distanceToGoal` - Already calculated in prep_extra_data.py line 281
- `quarter` - Need to check if this exists
- `half_seconds_remaining` - Need to check if this exists

### 4. Handle Missing Columns
If `down`, `quarter`, or `half_seconds_remaining` are missing, need to add them in `prep_extra_data.py`.

**Check the raw data schema first**, then add mapping if needed:
```python
# In map_column_names()
return df.rename({
    ...
    "down": "down",  # If it exists with different name
    "quarter": "quarter",  # If it exists with different name
})
```

### 5. Normalization Strategy
Game state features have different scales:
- `yardsToGo`: 1-99 (typically 1-20)
- `down`: 1-4
- `distanceToGoal`: 0-100
- `quarter`: 1-4
- `half_seconds_remaining`: 0-1800

**Options:**
1. **Let BatchNorm handle it** (current approach for other features)
   - BatchNorm1d in the model will normalize all 11 features together
   - Simplest approach

2. **Pre-normalize game state features**
   - Add in prep_extra_data.py
   - Min-max scaling: (x - min) / (max - min)
   - Z-score scaling: (x - mean) / std

**Recommendation:** Use BatchNorm1d (Option 1) - it's already working well for the existing features.

### 6. Testing Plan
1. **Data validation:** Check that game state features exist and have reasonable values
2. **Small-scale test:** Train on 10% sample to verify no shape errors
3. **Compare baseline:** Train model with and without game state features
4. **Ablation study:** Test different combinations of game state features

## Expected Performance Impact

**Hypothesis:** Game state features should improve model performance because:
1. **3rd down behavior:** Teams are more aggressive/conservative based on down
2. **Red zone:** Play calling changes dramatically near the goal line
3. **Yards to go:** Short-yardage vs long-yardage plays are fundamentally different
4. **Time context:** Quarter affects pace and play selection
5. **Time pressure:** Final minutes of each half see rushed play calling and prevent defense

**Metrics to track:**
- Validation loss (CrossEntropy)
- Mean Absolute Error on yards gained
- Performance stratified by game situation (3rd down, red zone, etc.)

## Potential Issues

### Issue 1: Missing Columns
**Symptom:** KeyError when loading features
**Solution:** Run notebook to check available columns, add missing columns to prep_extra_data.py

### Issue 2: Null Values
**Symptom:** NaN loss (similar to previous is_ball_carrier issue)
**Solution:** Add null checks in prep_extra_data.py filtering

### Issue 3: Scale Mismatch
**Symptom:** Training instability, slow convergence
**Solution:** Check BatchNorm is working, consider pre-normalization

### Issue 4: No Performance Improvement
**Possible reasons:**
- Model too small to use additional features
- Features already captured implicitly by player positions
- Need more training epochs to learn new features

**Debug approach:**
- Check feature correlations with yards gained
- Visualize attention weights for game state features
- Try larger model dimensions

## Notes on half_seconds_remaining
This feature captures time pressure within each half:
- Range: 0-1800 seconds (30 minutes per half)
- Critical for understanding urgency and play calling
- Teams behave very differently with <2 minutes remaining
- If not available in raw data, can be computed from `gameClock` + `quarter`:
  ```python
  # Quarters 1-2 are first half, 3-4 are second half
  # Each quarter is 15 minutes (900 seconds)
  half_seconds_remaining = calculate_from_game_clock_and_quarter(gameClock, quarter)
  ```

**Implementation:** Same as other game state features - add to feature list and broadcast to all players.

## Next Steps
1. ✅ Update notebook to explore game state features
2. ⬜ Run notebook to verify feature availability
3. ⬜ Check which game state columns exist in prepared data
4. ⬜ Update prep_extra_data.py if needed to add missing columns
5. ⬜ Update datasets.py to include game state features
6. ⬜ Update models.py feature_len
7. ⬜ Test on small sample
8. ⬜ Train full model and compare performance
