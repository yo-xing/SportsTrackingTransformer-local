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
    - {train,val,test}_features.parquet: Input features for models
    - {train,val,test}_targets.parquet: Yards gained labels
"""

import argparse
import shutil
from pathlib import Path

import polars as pl

INPUT_DATA_DIR = Path("/content/drive/MyDrive/NGS/NFL/REG/")
OUTPUT_DATA_DIR = Path("data/split_prepped_data_extra/")
DRIVE_DIR: Path | None = Path("/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_23entity_full") # Google Drive directory for caching (optional)

# Expected output files
OUTPUT_FILES = [
    "train_features.parquet",
    "train_targets.parquet",
    "test_features.parquet",
    "test_targets.parquet",
    "val_features.parquet",
    "val_targets.parquet",
]


def load_extra_data() -> pl.DataFrame:
    """
    Load all parquet files from extra_data directory structure.
    Expected structure: /content/drive/MyDrive/NGS/NFL/REG/Week XX/*.parquet

    Skips *_mirror.parquet files since we do our own mirroring.

    Returns:
        pl.DataFrame: Combined tracking data from all weeks.
    """
    parquet_files = [
        f for f in INPUT_DATA_DIR.glob("Week*/*.parquet")
        if "_mirror" not in f.name
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DATA_DIR}/Week*/")

    print(f"Found {len(parquet_files)} parquet files (excluding _mirror files)")

    dfs = []
    for f in parquet_files:
        df = pl.read_parquet(f)
        dfs.append(df)

    # Use vertical_relaxed to handle schema differences (e.g., Int64 vs Float64)
    return pl.concat(dfs, how="vertical_relaxed")


def map_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    Map extra_data column names to match the expected BDB 2024 format.

    Column mappings:
        gamekey -> gameId
        playid -> playId
        nfl_id -> nflId
        frame_id -> frameId
        display_name -> displayName
        team_abbr -> club
        yards_to_go -> yardsToGo
        yards_gained -> prePenaltyPlayResult
        vel_x -> vx (already Cartesian)
        vel_y -> vy (already Cartesian)
        vel -> s (speed magnitude)
    """
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


def filter_tracking_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter tracking data:
    - Remove football rows (possession_status == 'ball')
    - Keep only pass plays

    Args:
        df: Raw tracking data

    Returns:
        Filtered tracking data with only pass plays
    """
    og_len = len(df)

    # Filter out football rows
    df = df.filter(pl.col("possession_status") != "ball")
    print(f"Removed football rows: {og_len - len(df)} rows")

    # Keep only pass plays
    og_len = len(df)
    df = df.filter(pl.col("play_type") == "play_type_pass")
    print(f"Filtered to pass plays only: {og_len - len(df)} rows removed")

    return df


def identify_ball_carrier(df: pl.DataFrame) -> pl.DataFrame:
    """
    Identify the ball carrier for each play.

    Strategy: Find the offensive player closest to the football at the handoff frame.
    For plays without handoff event, use first_contact or ball_snap as fallback.

    Args:
        df: Tracking data with football positions

    Returns:
        DataFrame with ballCarrierId column added
    """
    # Load raw data to get football positions
    raw_df = load_extra_data()
    raw_df = map_column_names(raw_df)

    # Keep only pass plays (same as filter_tracking_data)
    raw_df = raw_df.filter(pl.col("play_type") == "play_type_pass")

    # Get football position at key events
    football_df = raw_df.filter(pl.col("possession_status") == "ball")

    # Priority events for identifying ball carrier
    priority_events = ["handoff", "first_contact", "ball_snap"]

    ball_carriers = []

    for (game_id, play_id), group in df.group_by(["gameId", "playId"]):
        # Get football positions for this play
        play_football = football_df.filter(
            (pl.col("gameId") == game_id) & (pl.col("playId") == play_id)
        )

        if len(play_football) == 0:
            continue

        # Find the event frame to use
        event_frame = None
        for event in priority_events:
            event_rows = play_football.filter(pl.col("event") == event)
            if len(event_rows) > 0:
                event_frame = event_rows["frameId"].min()
                break

        if event_frame is None:
            # Fallback to middle of play
            event_frame = play_football["frameId"].median()

        # Get football position at event frame
        fb_at_event = play_football.filter(pl.col("frameId") == event_frame)
        if len(fb_at_event) == 0:
            fb_at_event = play_football.filter(
                pl.col("frameId") == play_football["frameId"].min()
            )

        fb_x = fb_at_event["x"].item() if len(fb_at_event) > 0 else None
        fb_y = fb_at_event["y"].item() if len(fb_at_event) > 0 else None

        if fb_x is None or fb_y is None:
            continue

        # Get offensive players at event frame
        off_players = group.filter(
            (pl.col("frameId") == event_frame) &
            (pl.col("possession_status") == "off")
        )

        if len(off_players) == 0:
            continue

        # Find closest offensive player to football
        off_players = off_players.with_columns(
            dist=((pl.col("x") - fb_x) ** 2 + (pl.col("y") - fb_y) ** 2).sqrt()
        )

        closest = off_players.sort("dist").head(1)
        if len(closest) > 0:
            ball_carriers.append({
                "gameId": game_id,
                "playId": play_id,
                "ballCarrierId": closest["nflId"].item(),
            })

    if not ball_carriers:
        raise ValueError("Could not identify ball carrier for any plays")

    ball_carrier_df = pl.DataFrame(ball_carriers)
    print(f"Identified ball carriers for {len(ball_carrier_df)} plays")

    return df.join(ball_carrier_df, on=["gameId", "playId"], how="inner")


def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add derived features to match BDB 2024 preprocessing.

    Adds:
        - is_ball_carrier: Binary indicator
        - side: 1 for offense, -1 for defense
        - weight_Z, height_Z: Placeholder zeros (data not available)
        - ox, oy: Orientation vectors from 'o' angle
        - distanceToGoal: Distance to goal line
    """
    df = df.with_columns(
        is_ball_carrier=(pl.col("nflId") == pl.col("ballCarrierId")).cast(int),
        side=pl.when(pl.col("possession_status") == "off")
        .then(pl.lit(1))
        .otherwise(pl.lit(-1)),
        # Placeholder for missing player data
        weight_Z=pl.lit(0.0),
        height_Z=pl.lit(0.0),
    )

    # Convert orientation to unit vectors (already in degrees like BDB)
    # Convert from compass to unit-circle convention
    df = df.with_columns(
        o_rad=((pl.col("o") - 90) * -1) % 360,
    ).with_columns(
        ox=pl.col("o_rad").radians().cos(),
        oy=pl.col("o_rad").radians().sin(),
    ).drop("o_rad")

    # Calculate distance to goal
    # line_of_scrimmage is in field coordinates (0-100 where 0 is own goal line)
    # Assuming offense is moving right (x increasing), distance to goal = 100 - line_of_scrimmage
    df = df.with_columns(
        distanceToGoal=(100 - pl.col("line_of_scrimmage")).cast(pl.Float64)
    )

    return df


def determine_play_direction(df: pl.DataFrame) -> pl.DataFrame:
    """
    Determine play direction based on offensive player positions relative to line of scrimmage.

    If average offensive x > line_of_scrimmage, play direction is left (offense facing left goal)
    Otherwise, play direction is right

    Returns:
        DataFrame with playDirection column added
    """
    # Calculate average offensive x position at first frame of each play
    play_directions = (
        df.filter(pl.col("side") == 1)  # Offensive players
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

    return df.join(play_directions, on=["gameId", "playId"], how="left")


def standardize_tracking_directions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize play directions to always have offense moving left to right.
    """
    return df.with_columns(
        x=pl.when(pl.col("playDirection") == "right").then(pl.col("x")).otherwise(120 - pl.col("x")),
        y=pl.when(pl.col("playDirection") == "right").then(pl.col("y")).otherwise(53.3 - pl.col("y")),
        vx=pl.when(pl.col("playDirection") == "right").then(pl.col("vx")).otherwise(-1 * pl.col("vx")),
        vy=pl.when(pl.col("playDirection") == "right").then(pl.col("vy")).otherwise(-1 * pl.col("vy")),
        ox=pl.when(pl.col("playDirection") == "right").then(pl.col("ox")).otherwise(-1 * pl.col("ox")),
        oy=pl.when(pl.col("playDirection") == "right").then(pl.col("oy")).otherwise(-1 * pl.col("oy")),
    ).drop("playDirection")


def augment_mirror_tracking(df: pl.DataFrame) -> pl.DataFrame:
    """
    Augment data by mirroring the field across y-axis.
    """
    og_len = len(df)

    mirrored_df = df.clone().with_columns(
        y=53.3 - pl.col("y"),
        vy=-1 * pl.col("vy"),
        oy=-1 * pl.col("oy"),
        mirrored=pl.lit(True),
    )

    df = pl.concat(
        [
            df.with_columns(mirrored=pl.lit(False)),
            mirrored_df,
        ],
        how="vertical",
    )

    assert len(df) == og_len * 2, "Lost rows when mirroring tracking data"
    return df


def add_relative_positions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add relative position features anchored to ball carrier at first frame.
    """
    return (
        df.sort("frameId")
        .with_columns(
            anchor_x=pl.col("x").filter(pl.col("is_ball_carrier") == 1).first().over(["gameId", "playId", "mirrored"]),
            anchor_y=pl.col("y").filter(pl.col("is_ball_carrier") == 1).first().over(["gameId", "playId", "mirrored"]),
        )
        .with_columns(
            x_rel=pl.col("x") - pl.col("anchor_x"),
            y_rel=pl.col("y") - pl.col("anchor_y"),
        )
    )


def get_yards_gained_target_df(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Generate target dataframe for yards gained prediction.
    """
    MIN_YARDS = -10
    MAX_YARDS = 99

    # End events that indicate play completion
    END_EVENTS = ["tackle", "out_of_bounds", "touchdown", "qb_slide", "fumble"]

    plays_with_end = (
        df.filter(pl.col("event").is_in(END_EVENTS))
        .select(["gameId", "playId", "mirrored", "prePenaltyPlayResult"])
        .unique()
    )

    plays_with_end = plays_with_end.filter(pl.col("prePenaltyPlayResult").is_not_null())

    target_df = (
        df.select(["gameId", "playId", "mirrored", "frameId"])
        .unique()
        .join(plays_with_end, on=["gameId", "playId", "mirrored"], how="inner")
        .with_columns(
            yards_gained=pl.col("prePenaltyPlayResult").cast(pl.Float32),
        )
        .with_columns(
            yards_gained_class=(
                pl.col("yards_gained")
                .clip(MIN_YARDS, MAX_YARDS)
                .cast(pl.Int32) - MIN_YARDS
            ).cast(pl.Int64),
        )
        .drop("prePenaltyPlayResult")
    )

    og_play_count = len(df.select(["gameId", "playId"]).unique())
    df = df.join(
        target_df.select(["gameId", "playId", "mirrored"]).unique(),
        on=["gameId", "playId", "mirrored"],
        how="inner",
    )
    new_play_count = len(df.select(["gameId", "playId"]).unique())
    print(f"Lost {(og_play_count - new_play_count) / max(og_play_count, 1):.3%} plays when filtering for valid targets")

    unique_targets = target_df.select(["gameId", "playId", "mirrored", "yards_gained"]).unique()
    print(f"Yards gained stats: min={unique_targets['yards_gained'].min():.0f}, "
          f"max={unique_targets['yards_gained'].max():.0f}, "
          f"mean={unique_targets['yards_gained'].mean():.1f}, "
          f"median={unique_targets['yards_gained'].median():.1f}")

    return target_df, df


def split_train_test_val(tracking_df: pl.DataFrame, target_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Split data into train, validation, and test sets (70-15-15).
    """
    tracking_df = tracking_df.sort(["gameId", "playId", "mirrored", "frameId"])
    target_df = target_df.sort(["gameId", "playId", "mirrored", "frameId"])

    print(
        f"Total set: {tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{tracking_df.n_unique(['gameId', 'playId', 'mirrored', 'frameId'])} frames",
    )

    test_val_ids = tracking_df.select(["gameId", "playId"]).unique(maintain_order=True).sample(fraction=0.3, seed=42)
    train_tracking_df = tracking_df.join(test_val_ids, on=["gameId", "playId"], how="anti")
    train_tgt_df = target_df.join(test_val_ids, on=["gameId", "playId"], how="anti")
    print(
        f"Train set: {train_tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{train_tracking_df.n_unique(['gameId', 'playId', 'mirrored', 'frameId'])} frames",
    )

    test_ids = test_val_ids.sample(fraction=0.5, seed=42)
    test_tracking_df = tracking_df.join(test_ids, on=["gameId", "playId"], how="inner")
    test_tgt_df = target_df.join(test_ids, on=["gameId", "playId"], how="inner")
    print(
        f"Test set: {test_tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{test_tracking_df.n_unique(['gameId', 'playId', 'mirrored', 'frameId'])} frames",
    )

    val_ids = test_val_ids.join(test_ids, on=["gameId", "playId"], how="anti")
    val_tracking_df = tracking_df.join(val_ids, on=["gameId", "playId"], how="inner")
    val_tgt_df = target_df.join(val_ids, on=["gameId", "playId"], how="inner")
    print(
        f"Validation set: {val_tracking_df.n_unique(['gameId', 'playId', 'mirrored'])} plays,",
        f"{val_tracking_df.n_unique(['gameId', 'playId', 'mirrored', 'frameId'])} frames",
    )

    return {
        "train_features": train_tracking_df,
        "train_targets": train_tgt_df,
        "test_features": test_tracking_df,
        "test_targets": test_tgt_df,
        "val_features": val_tracking_df,
        "val_targets": val_tgt_df,
    }


def _check_drive_cache() -> bool:
    """Check if all output files exist in Google Drive cache."""
    if DRIVE_DIR is None:
        return False

    for filename in OUTPUT_FILES:
        drive_path = DRIVE_DIR / filename
        if not drive_path.exists():
            return False

    return True


def _load_from_drive() -> bool:
    """
    Load all output files from Google Drive cache.

    Returns True if successfully loaded from Drive, False otherwise.
    """
    if not _check_drive_cache():
        return False

    print(f"Found cached data in Drive: {DRIVE_DIR}")
    OUTPUT_DATA_DIR.mkdir(exist_ok=True, parents=True)

    for filename in OUTPUT_FILES:
        drive_path = DRIVE_DIR / filename
        local_path = OUTPUT_DATA_DIR / filename
        print(f"Copying {drive_path} -> {local_path}")
        shutil.copy2(drive_path, local_path)

    return True


def _save_to_drive():
    """Save all output files to Google Drive for caching."""
    if DRIVE_DIR is None:
        return

    print(f"\nSaving to Drive: {DRIVE_DIR}")
    DRIVE_DIR.mkdir(exist_ok=True, parents=True)

    for filename in OUTPUT_FILES:
        local_path = OUTPUT_DATA_DIR / filename
        drive_path = DRIVE_DIR / filename
        if local_path.exists():
            print(f"Copying {local_path} -> {drive_path}")
            shutil.copy2(local_path, drive_path)


def main(output_dir: Path = OUTPUT_DATA_DIR, drive_dir: Path | None = None):
    """Main execution function for extra data preparation."""
    global OUTPUT_DATA_DIR, DRIVE_DIR
    OUTPUT_DATA_DIR = output_dir
    DRIVE_DIR = drive_dir

    if DRIVE_DIR is not None:
        print(f"Google Drive caching enabled: {DRIVE_DIR}")

        # Check if all files exist in Drive cache
        if _load_from_drive():
            print("All files loaded from Drive cache, skipping computation")
            return

    print("Loading extra data...")
    df = load_extra_data()
    print(f"Loaded {len(df)} rows")

    print("\nMapping column names...")
    df = map_column_names(df)

    print("\nFiltering tracking data...")
    df = filter_tracking_data(df)

    print("\nIdentifying ball carriers...")
    df = identify_ball_carrier(df)

    print("\nAdding derived features...")
    df = add_derived_features(df)

    print("\nDetermining play directions...")
    df = determine_play_direction(df)

    print("\nStandardizing directions...")
    df = standardize_tracking_directions(df)

    print("\nAugmenting with mirrored data...")
    df = augment_mirror_tracking(df)

    print("\nAdding relative positions...")
    df = add_relative_positions(df)

    print("\nGenerating targets...")
    target_df, df = get_yards_gained_target_df(df)

    print("\nSplitting data...")
    split_dfs = split_train_test_val(df, target_df)

    OUTPUT_DATA_DIR.mkdir(exist_ok=True, parents=True)

    for key, split_df in split_dfs.items():
        sort_keys = ["gameId", "playId", "mirrored", "frameId"]
        out_path = OUTPUT_DATA_DIR / f"{key}.parquet"
        split_df.sort(sort_keys).write_parquet(out_path)
        print(f"Saved {out_path}")

    # Save to Drive for future runs
    _save_to_drive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare extra tracking data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DATA_DIR,
        help="Directory to output preprocessed parquet files",
    )
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=DRIVE_DIR,
        help="Google Drive directory for caching (e.g., /content/drive/MyDrive/prepped_data_extra)",
    )
    args = parser.parse_args()

    main(output_dir=args.output_dir, drive_dir=args.drive_dir)
