"""
Data Preparation Module for Extra NFL Tracking Data (Axially format)

This module adapts the prep_data.py pipeline for the extra_data format which has:
- Different column naming (snake_case vs camelCase)
- Velocity already in Cartesian format (vel_x, vel_y)
- Combined tracking/play data in single parquet files
- No separate player height/weight data
- Ball carrier identified via proximity to football at handoff

Pipeline Flow:
    Raw Parquet Data -> Column Mapping -> Filtering -> Ball Carrier ID ->
    Feature Engineering -> Standardization -> Augmentation -> Train/Val/Test Split

Output:
    Creates 6 files in data/split_prepped_data_extra/:
    - {train,val,test}_features.parquet
    - {train,val,test}_targets.parquet
"""

import argparse
import shutil
from pathlib import Path
import polars as pl

# =====================
# CONFIG
# =====================

INPUT_DATA_DIR = Path("/content/drive/MyDrive/NGS/NFL/REG/")
OUTPUT_DATA_DIR = Path("data/split_prepped_data_extra/")
DRIVE_DIR: Path | None = Path(
    "/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache"
)

OUTPUT_FILES = [
    "train_features.parquet",
    "train_targets.parquet",
    "test_features.parquet",
    "test_targets.parquet",
    "val_features.parquet",
    "val_targets.parquet",
]

ACCEPTED_PLAY_TYPES = ["play_type_pass", "play_type_rush", "play_type_sack"]
WEEKS_TO_READ = ["06", "07"]

MIN_YARDS = -10
MAX_YARDS = 99

# =====================
# LOAD DATA
# =====================

def load_extra_data() -> pl.DataFrame:
    parquet_files = []
    for wk in WEEKS_TO_READ:
        parquet_files.extend(INPUT_DATA_DIR.glob(f"Week {wk}/*.parquet"))
        parquet_files.extend(INPUT_DATA_DIR.glob(f"Week{wk}/*.parquet"))

    parquet_files = [f for f in parquet_files if "_mirror" not in f.name]

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for Weeks {WEEKS_TO_READ} under {INPUT_DATA_DIR}"
        )

    print(f"Found {len(parquet_files)} parquet files from Weeks {WEEKS_TO_READ}")
    return pl.concat([pl.read_parquet(f) for f in parquet_files], how="vertical_relaxed")

# =====================
# COLUMN MAPPING
# =====================

def map_column_names(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({
        "gamekey": "gameId",
        "playid": "playId",
        "nfl_id": "nflId",
        "frame_id": "frameId",
        "display_name": "displayName",
        "team_abbr": "club",
        "yards_to_go": "yardsToGo",
        "yards_gained": "prePenaltyPlayResult",
        "vel_x": "vx",
        "vel_y": "vy",
        "vel": "s",
        "roster_position": "position",
    })

# =====================
# FILTERING
# =====================

def filter_tracking_data(df: pl.DataFrame) -> pl.DataFrame:
    og_len = len(df)
    df = df.filter(pl.col("possession_status") != "ball")
    print(f"Removed football rows: {og_len - len(df)}")

    og_len = len(df)
    df = df.filter(pl.col("play_type").is_in(ACCEPTED_PLAY_TYPES))
    print(f"Filtered to accepted play types: {og_len - len(df)}")

    return df

# =====================
# BALL CARRIER ID
# =====================

def identify_ball_carrier(df: pl.DataFrame) -> pl.DataFrame:
    raw_df = map_column_names(load_extra_data())
    raw_df = raw_df.filter(pl.col("play_type").is_in(ACCEPTED_PLAY_TYPES))
    football_df = raw_df.filter(pl.col("possession_status") == "ball")

    priority_events = ["handoff", "first_contact", "ball_snap"]
    records = []

    for (gid, pid), group in df.group_by(["gameId", "playId"]):
        fb = football_df.filter(
            (pl.col("gameId") == gid) & (pl.col("playId") == pid)
        )
        if len(fb) == 0:
            continue

        event_frame = None
        for ev in priority_events:
            rows = fb.filter(pl.col("event") == ev)
            if len(rows) > 0:
                event_frame = rows["frameId"].min()
                break
        if event_frame is None:
            event_frame = fb["frameId"].median()

        fb_at = fb.filter(pl.col("frameId") == event_frame)
        if len(fb_at) == 0:
            continue

        fx, fy = fb_at["x"].item(), fb_at["y"].item()

        off = group.filter(
            (pl.col("frameId") == event_frame) &
            (pl.col("possession_status") == "off")
        )
        if len(off) == 0:
            continue

        off = off.with_columns(
            dist=((pl.col("x") - fx)**2 + (pl.col("y") - fy)**2).sqrt()
        )
        closest = off.sort("dist").head(1)

        records.append({
            "gameId": gid,
            "playId": pid,
            "ballCarrierId": closest["nflId"].item(),
        })

    ball_carrier_df = pl.DataFrame(records)

    # IMPORTANT: left join so plays are NOT filtered out
    return df.join(ball_carrier_df, on=["gameId", "playId"], how="left")

# =====================
# FEATURE ENGINEERING
# =====================

def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        is_ball_carrier=(pl.col("nflId") == pl.col("ballCarrierId"))
        .fill_null(False)
        .cast(int),
        side=pl.when(pl.col("possession_status") == "off").then(1).otherwise(-1),
        weight_Z=pl.lit(0.0),
        height_Z=pl.lit(0.0),
    )

    df = df.with_columns(
        o_rad=((pl.col("o") - 90) * -1) % 360,
    ).with_columns(
        ox=pl.col("o_rad").radians().cos(),
        oy=pl.col("o_rad").radians().sin(),
    ).drop("o_rad")

    df = df.with_columns(
        distanceToGoal=(100 - pl.col("line_of_scrimmage")).cast(pl.Float64)
    )

    return df

# =====================
# PLAY DIRECTION
# =====================

def determine_play_direction(df: pl.DataFrame) -> pl.DataFrame:
    play_dir = (
        df.filter(pl.col("side") == 1)
        .group_by(["gameId", "playId"])
        .agg([
            pl.col("x").filter(pl.col("frameId") == pl.col("frameId").min()).mean().alias("avg_off_x"),
            pl.col("line_of_scrimmage").first().alias("los"),
        ])
        .with_columns(
            playDirection=pl.when(pl.col("avg_off_x") > pl.col("los"))
            .then(pl.lit("left"))
            .otherwise(pl.lit("right"))
        )
        .select(["gameId", "playId", "playDirection"])
    )
    return df.join(play_dir, on=["gameId", "playId"], how="left")

def standardize_tracking_directions(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        x=pl.when(pl.col("playDirection") == "right").then(pl.col("x")).otherwise(120 - pl.col("x")),
        y=pl.when(pl.col("playDirection") == "right").then(pl.col("y")).otherwise(53.3 - pl.col("y")),
        vx=pl.when(pl.col("playDirection") == "right").then(pl.col("vx")).otherwise(-pl.col("vx")),
        vy=pl.when(pl.col("playDirection") == "right").then(pl.col("vy")).otherwise(-pl.col("vy")),
        ox=pl.when(pl.col("playDirection") == "right").then(pl.col("ox")).otherwise(-pl.col("ox")),
        oy=pl.when(pl.col("playDirection") == "right").then(pl.col("oy")).otherwise(-pl.col("oy")),
    ).drop("playDirection")

# =====================
# MIRROR + RELATIVE POSITIONS
# =====================

def augment_mirror_tracking(df: pl.DataFrame) -> pl.DataFrame:
    mirrored = df.clone().with_columns(
        y=53.3 - pl.col("y"),
        vy=-pl.col("vy"),
        oy=-pl.col("oy"),
        mirrored=pl.lit(True),
    )
    return pl.concat(
        [df.with_columns(mirrored=pl.lit(False)), mirrored],
        how="vertical",
    )

def add_relative_positions(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort("frameId")
        .with_columns(
            anchor_x=pl.coalesce([
                pl.col("x").filter(pl.col("is_ball_carrier") == 1).first()
                .over(["gameId", "playId", "mirrored"]),
                pl.col("x").filter(pl.col("side") == 1).first()
                .over(["gameId", "playId", "mirrored"]),
            ]),
            anchor_y=pl.coalesce([
                pl.col("y").filter(pl.col("is_ball_carrier") == 1).first()
                .over(["gameId", "playId", "mirrored"]),
                pl.col("y").filter(pl.col("side") == 1).first()
                .over(["gameId", "playId", "mirrored"]),
            ]),
        )
        .with_columns(
            x_rel=pl.col("x") - pl.col("anchor_x"),
            y_rel=pl.col("y") - pl.col("anchor_y"),
        )
    )

# =====================
# TARGETS
# =====================

def get_yards_gained_target_df(df: pl.DataFrame):
    END_EVENTS = ["tackle", "out_of_bounds", "touchdown", "qb_slide", "fumble"]

    plays = (
        df.filter(pl.col("event").is_in(END_EVENTS))
        .select(["gameId", "playId", "mirrored", "prePenaltyPlayResult"])
        .unique()
        .filter(pl.col("prePenaltyPlayResult").is_not_null())
    )

    target_df = (
        df.select(["gameId", "playId", "mirrored", "frameId"])
        .unique()
        .join(plays, on=["gameId", "playId", "mirrored"], how="inner")
        .with_columns(
            yards_gained=pl.col("prePenaltyPlayResult").cast(pl.Float32),
            yards_gained_class=(
                pl.col("prePenaltyPlayResult")
                .clip(MIN_YARDS, MAX_YARDS)
                .cast(pl.Int32) - MIN_YARDS
            ).cast(pl.Int64),
        )
        .drop("prePenaltyPlayResult")
    )

    df = df.join(
        target_df.select(["gameId", "playId", "mirrored"]).unique(),
        on=["gameId", "playId", "mirrored"],
        how="inner",
    )

    return target_df, df

# =====================
# SPLIT + SAVE
# =====================

def split_train_test_val(tracking_df, target_df):
    ids = tracking_df.select(["gameId", "playId"]).unique()
    test_val = ids.sample(fraction=0.3, seed=42)
    test = test_val.sample(fraction=0.5, seed=42)
    val = test_val.join(test, on=["gameId", "playId"], how="anti")

    return {
        "train_features": tracking_df.join(test_val, on=["gameId", "playId"], how="anti"),
        "train_targets": target_df.join(test_val, on=["gameId", "playId"], how="anti"),
        "test_features": tracking_df.join(test, on=["gameId", "playId"], how="inner"),
        "test_targets": target_df.join(test, on=["gameId", "playId"], how="inner"),
        "val_features": tracking_df.join(val, on=["gameId", "playId"], how="inner"),
        "val_targets": target_df.join(val, on=["gameId", "playId"], how="inner"),
    }

# =====================
# MAIN
# =====================

def main(output_dir=OUTPUT_DATA_DIR):
    df = load_extra_data()
    df = map_column_names(df)
    df = filter_tracking_data(df)
    df = identify_ball_carrier(df)
    df = add_derived_features(df)
    df = determine_play_direction(df)
    df = standardize_tracking_directions(df)
    df = augment_mirror_tracking(df)
    df = add_relative_positions(df)

    target_df, df = get_yards_gained_target_df(df)
    splits = split_train_test_val(df, target_df)

    output_dir.mkdir(exist_ok=True, parents=True)
    for k, v in splits.items():
        v.sort(["gameId", "playId", "mirrored", "frameId"]).write_parquet(
            output_dir / f"{k}.parquet"
        )

if __name__ == "__main__":
    main()
