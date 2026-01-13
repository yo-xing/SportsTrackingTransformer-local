#!/usr/bin/env python3
"""Quick check for Inf/NaN values in prepared data."""
import polars as pl
from pathlib import Path
import sys

# Check if data exists
data_dir = Path("data/split_prepped_data_extra_full18weeks")
if not data_dir.exists():
    print(f"❌ Data directory not found: {data_dir}")
    print("   You need to run data preparation first.")
    sys.exit(1)

print("Checking for Inf/NaN values in prepared data...\n")

issues_found = False

for split in ["train", "val", "test"]:
    features_file = data_dir / f"{split}_features.parquet"
    targets_file = data_dir / f"{split}_targets.parquet"
    
    if not features_file.exists():
        print(f"⚠️  {split}_features.parquet not found")
        continue
    
    # Read features
    df = pl.read_parquet(features_file)
    
    # Check the actual model input features
    model_features = ["x_rel", "y_rel", "vx", "vy", "side", "is_ball_carrier"]
    
    print(f"📊 {split.upper()} split:")
    print(f"   Total rows: {len(df):,}")
    
    for col in model_features:
        if col not in df.columns:
            print(f"   ❌ Column '{col}' missing!")
            issues_found = True
            continue
            
        # Check for nulls
        null_count = df[col].null_count()
        if null_count > 0:
            print(f"   ⚠️  {col}: {null_count:,} null values ({null_count/len(df):.2%})")
            issues_found = True
        
        # Check for inf values (only for numeric columns)
        if col in ["x_rel", "y_rel", "vx", "vy", "side", "is_ball_carrier"]:
            inf_count = df.filter(~pl.col(col).is_finite()).height
            if inf_count > 0:
                print(f"   ⚠️  {col}: {inf_count:,} Inf values ({inf_count/len(df):.2%})")
                issues_found = True
    
    # Check targets
    if targets_file.exists():
        tgt_df = pl.read_parquet(targets_file)
        null_targets = tgt_df["yards_gained_class"].null_count()
        if null_targets > 0:
            print(f"   ⚠️  targets: {null_targets:,} null values")
            issues_found = True
        
        # Check if target values are in valid range [0, 109]
        invalid_targets = tgt_df.filter(
            (pl.col("yards_gained_class") < 0) | 
            (pl.col("yards_gained_class") > 109)
        ).height
        if invalid_targets > 0:
            print(f"   ⚠️  targets: {invalid_targets:,} out of range [0, 109]")
            issues_found = True
    
    if not issues_found:
        print(f"   ✅ All features clean")
    print()

if issues_found:
    print("\n❌ ISSUES FOUND: Data needs to be regenerated with new filtering!")
    print("   Run: uv run python run_transformer_only.py")
    sys.exit(1)
else:
    print("\n✅ No issues found! Data is clean.")
    print("   You can skip data prep: uv run python run_transformer_only.py --skip-prep --skip-precompute")
    sys.exit(0)
