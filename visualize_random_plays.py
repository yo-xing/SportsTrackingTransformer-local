"""
Randomly sample and visualize 5 plays of each type from the 23-entity-football model.
Saves HTML animations and PNG graphs to ./visualizations_random/ folder.
"""

import re
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = Path('./visualizations_random')
OUTPUT_DIR.mkdir(exist_ok=True)

# Paths (adjust for your environment)
MODELS_DIR = Path('/content/drive/MyDrive/SportsTrackingTransformer/models_norm_football')
NGS_DATA_DIR = Path('/content/drive/MyDrive/NGS/NFL/REG/')
PREPPED_DATA_PATH = Path('/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_gamestate_norm')

PLAY_TYPES = ['play_type_pass', 'play_type_rush', 'play_type_sack']
MIN_YARDS = -10

print("="*80)
print("🏈 VISUALIZING RANDOM PLAYS FROM 23-ENTITY-FOOTBALL MODEL")
print("="*80)
print()

def get_epoch_val_loss_from_ckpt(ckpt_path: Path):
    """Extract epoch and val_loss from checkpoint filename."""
    val_loss_pattern = re.compile(r"epoch=(\d+)-val_loss=([\d\.]+)")
    match = val_loss_pattern.search(ckpt_path.name)
    if match:
        try:
            return int(match.group(1)), float(match.group(2).rstrip("."))
        except:
            return -1, float('inf')
    return -1, float('inf')

def find_best_model(models_root: Path):
    """Find the best checkpoint by lowest val_loss."""
    model_dir = models_root / 'transformer'
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return None

    best_ckpt = None
    best_val_loss = float('inf')

    for config_dir in model_dir.iterdir():
        if not config_dir.is_dir() or config_dir.name == 'best_models':
            continue

        ckpt_dir = config_dir / 'checkpoints'
        if not ckpt_dir.exists():
            continue

        ckpts = list(ckpt_dir.glob('*.ckpt'))
        for ckpt in ckpts:
            _, val_loss = get_epoch_val_loss_from_ckpt(ckpt)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = ckpt

    if best_ckpt:
        results_path = best_ckpt.with_suffix('.results.parquet')
        if results_path.exists():
            return results_path

    return None

def animate_play_with_predictions(
    tracking_df: pl.DataFrame,
    test_results: pl.DataFrame,
    gameId: int,
    playId: int,
):
    """
    Animate a play with expected yards predictions shown as horizontal lines.
    """
    # Filter tracking data for this play
    mvmt_df = (
        tracking_df.filter(
            (pl.col("gameId") == gameId) & (pl.col("playId") == playId)
        )
        .to_pandas()
        .round(2)
    )

    # Filter test results for this play
    results_df = test_results.filter(
        (pl.col("gameId") == gameId) & (pl.col("playId") == playId)
    ).to_pandas()

    if len(mvmt_df) == 0:
        print(f"No tracking data found for gameId={gameId}, playId={playId}")
        return None

    if len(results_df) == 0:
        print(f"No results data found for gameId={gameId}, playId={playId}")
        return None

    # Get play metadata
    down = mvmt_df["down"].iloc[0] if "down" in mvmt_df.columns else None
    yards_to_go = mvmt_df["yardsToGo"].iloc[0] if "yardsToGo" in mvmt_df.columns else None
    quarter = mvmt_df["quarter"].iloc[0] if "quarter" in mvmt_df.columns else None
    play_type = mvmt_df["play_type"].iloc[0] if "play_type" in mvmt_df.columns else "Unknown"
    actual_yards = results_df["yards_gained"].iloc[0]

    # Get the football's position at the first frame - this is our LOS
    ball_first_frame = mvmt_df[mvmt_df["side"] == 0].iloc[0]
    los = ball_first_frame["x"]

    # Convert all x positions to be relative to LOS
    mvmt_df["x_rel"] = mvmt_df["x"] - los

    # Clean up play type
    if play_type and str(play_type).startswith("play_type_"):
        play_type = play_type.replace("play_type_", "").replace("_", " ").title()

    mvmt_df["side"] = mvmt_df["side"].replace({1: "OFF", -1: "DEF", 0: "BALL"})

    # Get clubs
    off_club = mvmt_df.loc[mvmt_df["side"] == "OFF", "club"].values[0] if len(mvmt_df[mvmt_df["side"] == "OFF"]) > 0 else "OFF"
    def_club = mvmt_df.loc[mvmt_df["side"] == "DEF", "club"].values[0] if len(mvmt_df[mvmt_df["side"] == "DEF"]) > 0 else "DEF"

    mvmt_y_min = mvmt_df["x_rel"].min()
    mvmt_y_max = mvmt_df["x_rel"].max()

    # Prepare movement data
    mvmt_df["size"] = 2
    mvmt_df["text_color"] = "black"
    mvmt_df["symbol"] = "player"
    mvmt_df.loc[mvmt_df["is_ball_carrier"] == 1, "symbol"] = "ball_carrier"
    mvmt_df.loc[mvmt_df["is_ball_carrier"] == 1, "size"] = 3

    symbol_map = {"player": "circle", "ball_carrier": "hexagon"}

    # Create frameId_event for animation
    mvmt_df["frameId_event"] = mvmt_df["frameId"].astype(str) + mvmt_df["event"].fillna("").astype(str).apply(
        lambda x: f"_{x}" if len(x) > 0 else x
    )

    # Hover data
    hover_data = {
        "displayName": True,
        "club": False,
        "side": False,
        "jersey_number": False,
        "is_ball_carrier": True,
        "symbol": False,
        "frameId": False,
        "x_rel": True,
        "y": True,
        "vx": True,
        "vy": True,
        "size": False,
    }

    # Field dimensions
    X_LEFT = 0
    X_MIDDLE = 160 / 6.0
    X_RIGHT = 160 / 3.0

    fig = px.scatter(
        data_frame=mvmt_df,
        x="y",
        y="x_rel",
        animation_frame="frameId_event",
        animation_group="nflId",
        hover_name="displayName",
        hover_data=hover_data,
        text="jersey_number",
        width=1000,
        height=900,
        range_x=[X_LEFT - 2, X_RIGHT + 2],
        size="size",
        size_max=15,
        color="side",
        color_discrete_map={"OFF": "#39FF14", "DEF": "#FF69B4", "BALL": "brown"},
        opacity=0.9,
        symbol="symbol",
        symbol_map=symbol_map,
    )

    # LOS is now at 0 (relative coordinates)
    fig.add_shape(
        type="line",
        x0=X_LEFT,
        y0=0,
        x1=X_RIGHT,
        y1=0,
        line=dict(color="rgba(137, 207, 240, 0.5)", width=3, dash="dash"),
    )

    # Actual yards is relative to 0
    fig.add_shape(
        type="line",
        x0=X_LEFT,
        y0=actual_yards,
        x1=X_RIGHT,
        y1=actual_yards,
        line=dict(color="rgba(0, 255, 0, 0.5)", width=3, dash="solid"),
    )

    # Add yards to go line if available
    if yards_to_go is not None and not pd.isna(yards_to_go):
        fig.add_shape(
            type="line",
            x0=X_LEFT,
            y0=yards_to_go,
            x1=X_RIGHT,
            y1=yards_to_go,
            line=dict(color="rgba(255, 255, 0, 0.3)", width=3, dash="dash"),
        )

    # Add animated prediction lines
    frames = []
    for frame in fig.frames:
        frame_id_str = frame.name
        frame_id = int(frame_id_str.split("_")[0])

        if frame_id in results_df["frameId"].values:
            expected_yards = results_df[results_df["frameId"] == frame_id]["expected_yards"].iloc[0]

            shapes_list = [
                dict(
                    type="line",
                    x0=X_LEFT,
                    y0=0,
                    x1=X_RIGHT,
                    y1=0,
                    line=dict(color="rgba(137, 207, 240, 0.5)", width=3, dash="dash"),
                ),
                dict(
                    type="line",
                    x0=X_LEFT,
                    y0=actual_yards,
                    x1=X_RIGHT,
                    y1=actual_yards,
                    line=dict(color="rgba(0, 255, 0, 0.5)", width=3, dash="solid"),
                ),
            ]

            if yards_to_go is not None and not pd.isna(yards_to_go):
                shapes_list.append(
                    dict(
                        type="line",
                        x0=X_LEFT,
                        y0=yards_to_go,
                        x1=X_RIGHT,
                        y1=yards_to_go,
                        line=dict(color="rgba(255, 255, 0, 0.3)", width=3, dash="dash"),
                    )
                )

            shapes_list.append(
                dict(
                    type="line",
                    x0=X_LEFT,
                    y0=expected_yards,
                    x1=X_RIGHT,
                    y1=expected_yards,
                    line=dict(color="rgba(255, 215, 0, 0.8)", width=4, dash="dot"),
                )
            )

            frame.layout.shapes = shapes_list

        frames.append(frame)

    fig.frames = frames

    # Set play speed
    frame_duration = 100
    for button in fig.layout.updatemenus[0].buttons:
        button["args"][1]["frame"]["duration"] = frame_duration

    # Set aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # Background color
    fig.update_layout(paper_bgcolor="#333333", plot_bgcolor="#363636", font_color="white", font_size=14)

    # Turn off axis
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)

    # Grid line thickness
    fig.update_yaxes(
        showgrid=True,
        gridwidth=3,
        gridcolor="rgba(237, 234, 222, 0.1)",
        linewidth=0,
        linecolor="rgba(0, 0, 0, 0.01)",
        mirror=True,
        showticklabels=False,
    )

    # Set y axis range
    fig.update_yaxes(range=[mvmt_y_min - 5, mvmt_y_max + 5])

    # Text size
    fig.update_layout(uniformtext_minsize=2, uniformtext_mode="hide")

    # Hide legend
    fig.update_layout(showlegend=False)

    # Text color of jersey numbers
    fig.update_traces(textfont=dict(family="Tahoma", size=12, color=mvmt_df["text_color"]))
    fig.update_traces(marker_line_width=0)

    # Hide x and y labels
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")

    # Add relative yard markers
    y_range_start = int(mvmt_y_min // 5) * 5
    y_range_end = int(mvmt_y_max // 5) * 5 + 5

    for y_loc in range(y_range_start, y_range_end + 1, 5):
        if y_loc == 0:
            fig.add_annotation(
                x=X_LEFT + 2,
                y=y_loc,
                text="LOS",
                showarrow=False,
                font=dict(color="rgba(137, 207, 240, 0.8)", size=20),
                opacity=0.8,
            )
        elif y_loc % 10 == 0:
            fig.add_annotation(
                x=X_LEFT + 2,
                y=y_loc,
                text=f"{y_loc:+d}",
                showarrow=False,
                font=dict(color="white", size=16),
                opacity=0.3,
            )

    # Add legend for lines
    fig.add_annotation(
        x=X_LEFT + 2,
        y=mvmt_y_max + 3,
        text="<span style='color:rgba(137,207,240,0.5)'>━━</span> LOS | <span style='color:rgba(255,255,0,0.3)'>━━</span> 1st Down | <span style='color:rgba(0,255,0,0.5)'>━━</span> Actual | <span style='color:rgba(255,215,0,0.8)'>┄┄</span> Predicted",
        showarrow=False,
        font=dict(color="white", size=12),
        opacity=0.9,
        xanchor="left",
    )

    # Set title
    fig.update_layout(
        title=f"Game {gameId}, Play {playId} | {off_club} vs {def_club} | {play_type} | Down: {down} | YTG: {yards_to_go} | Actual: {actual_yards:.1f} yds",
        font_size=12,
        title_x=0.5,
        title_y=0.98,
    )

    return fig

def visualize_play_v3(test_results: pl.DataFrame, game_id: int, play_id: int, output_path: Path, prepped_data_path: Path):
    """Create matplotlib graph showing predicted vs actual yards with event markers."""

    play_data = test_results.filter(
        (pl.col("gameId") == game_id) &
        (pl.col("playId") == play_id)
    ).sort("frameId")

    if len(play_data) == 0:
        print(f"  ⚠️  No data found for gameId={game_id}, playId={play_id}")
        return None

    play_df = play_data.to_pandas()
    actual_yards = play_df["yards_gained"].iloc[0]

    # Load the original prepped data to get play type, metadata, and events
    play_type = "Unknown"
    play_desc = "Description not available"
    events_df = None

    try:
        test_data_path = prepped_data_path / "test_features.parquet"
        if test_data_path.exists():
            full_data = pl.read_parquet(test_data_path)

            play_info = full_data.filter(
                (pl.col("gameId") == game_id) &
                (pl.col("playId") == play_id)
            )

            if len(play_info) > 0:
                if "play_type" in play_info.columns:
                    play_type = play_info["play_type"][0]
                    if play_type and play_type.startswith("play_type_"):
                        play_type = play_type.replace("play_type_", "").replace("_", " ").title()

                desc_parts = []
                if "down" in play_info.columns and "yardsToGo" in play_info.columns:
                    down = play_info["down"][0]
                    yards_to_go = play_info["yardsToGo"][0]
                    if down is not None and yards_to_go is not None:
                        desc_parts.append(f"{int(down)} & {int(yards_to_go)}")

                if "quarter" in play_info.columns:
                    quarter = play_info["quarter"][0]
                    if quarter is not None:
                        desc_parts.append(f"Q{int(quarter)}")

                if desc_parts:
                    play_desc = " | ".join(desc_parts)

                if "event" in play_info.columns and "frameId" in play_info.columns:
                    events_data = play_info.filter(
                        pl.col("event").is_not_null() &
                        (pl.col("event") != "None") &
                        (pl.col("event") != "")
                    ).select(["frameId", "event"]).unique()

                    if len(events_data) > 0:
                        events_df = events_data.to_pandas()

    except Exception as e:
        print(f"  ⚠️  Could not load play metadata: {e}")

    # Truncate description if too long
    if len(str(play_desc)) > 80:
        play_desc_short = str(play_desc)[:77] + "..."
    else:
        play_desc_short = str(play_desc)

    # Create single plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot predicted yards (blue line)
    ax.plot(play_df["frameId"], play_df["expected_yards"],
            'b-o', linewidth=2.5, markersize=7, label='Predicted Yards', alpha=0.8)

    # Plot actual yards (horizontal red line)
    ax.axhline(y=actual_yards, color='red', linestyle='--', linewidth=3,
               label=f'Actual Yards: {actual_yards:.1f}', alpha=0.9)

    # Add event markers if available
    if events_df is not None and len(events_df) > 0:
        # Get y-axis limits for positioning
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        for idx, event_row in events_df.iterrows():
            frame_id = event_row["frameId"]
            event_name = event_row["event"]

            # Draw vertical line for event
            ax.axvline(x=frame_id, color='orange', linestyle=':', alpha=0.6, linewidth=2)

            # Add event label at the top
            ax.text(frame_id, y_max - (y_range * 0.05), event_name,
                    rotation=90, verticalalignment='top', horizontalalignment='right',
                    fontsize=9, color='orange', fontweight='bold')

    ax.set_xlabel("Frame ID", fontsize=14, fontweight='bold')
    ax.set_ylabel("Yards Gained", fontsize=14, fontweight='bold')
    ax.set_title(f"Expected vs Actual Yards Gained\nGame {game_id}, Play {play_id} | {play_type} | {play_desc_short}",
                  fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path

print("Step 1: Finding best model results...")
results_path = find_best_model(MODELS_DIR)

if not results_path:
    print("❌ No results file found!")
    exit(1)

print(f"✓ Found results: {results_path.name}")

print("\nStep 2: Loading results...")
results_df = pl.read_parquet(results_path)
print(f"✓ Loaded {len(results_df):,} predictions")

# Filter to test set
split_col = 'dataset_split' if 'dataset_split' in results_df.columns else 'split'
test_df = results_df.filter(pl.col(split_col) == 'test')
print(f"✓ Test set: {len(test_df):,} predictions")

print("\nStep 3: Loading NGS metadata...")
parquet_files = []
for week_dir in NGS_DATA_DIR.glob("Week*"):
    if week_dir.is_dir():
        parquet_files.extend([f for f in week_dir.glob("*.parquet") if "_mirror" not in f.name])

print(f"Found {len(parquet_files)} parquet files")
ngs_dfs = []
for i, f in enumerate(parquet_files[:100]):
    if i % 25 == 0:
        print(f"  Loading file {i+1}/100...")
    ngs_dfs.append(pl.read_parquet(f))

ngs_df = pl.concat(ngs_dfs, how="vertical_relaxed")

# Normalize column names
rename_map = {'gamekey': 'gameId', 'playid': 'playId', 'frame_id': 'frameId'}
for old, new in rename_map.items():
    if old in ngs_df.columns:
        ngs_df = ngs_df.rename({old: new})

ngs_metadata = ngs_df.select(['gameId', 'playId', 'frameId', 'play_type', 'event']).unique()
print(f"✓ Loaded {len(ngs_metadata):,} metadata records")

print("\nStep 4: Joining predictions with metadata...")
# Handle mirrored plays
test_orig = test_df.filter(pl.col('mirrored') == False)
joined_orig = test_orig.join(ngs_metadata, on=['gameId', 'playId', 'frameId'], how='left')

test_mirror = test_df.filter(pl.col('mirrored') == True)
if len(test_mirror) > 0:
    play_meta = ngs_metadata.select(['gameId', 'playId', 'play_type']).unique()
    joined_mirror = test_mirror.join(play_meta, on=['gameId', 'playId'], how='left')
    joined_mirror = joined_mirror.with_columns(pl.lit(None).alias('event'))
    joined = pl.concat([joined_orig, joined_mirror], how="vertical_relaxed")
else:
    joined = joined_orig

print(f"✓ Joined: {len(joined):,} predictions with metadata")

print("\nStep 5: Computing MAE per play...")
# Note: results already contain 'expected_yards' (weighted average from softmax)
# Just need to add true_yards from the class labels
joined = joined.with_columns([
    (pl.col('yards_gained_class') + MIN_YARDS).alias('true_yards')
])

# Compute absolute error using the expected_yards (weighted average) from results
joined = joined.with_columns(
    (pl.col('true_yards') - pl.col('expected_yards')).abs().alias('abs_error')
)

# Get play-level MAE (average across all frames in the play)
play_mae = joined.group_by(['gameId', 'playId', 'play_type']).agg([
    pl.col('abs_error').mean().alias('play_mae'),
    pl.col('true_yards').first().alias('yards_gained'),
    pl.col('mirrored').first().alias('mirrored')
])

print(f"✓ Computed MAE for {len(play_mae):,} plays")

print("\nStep 6: Randomly sampling 5 plays per type...")
random_plays = []

for play_type in PLAY_TYPES:
    play_type_clean = play_type.replace('play_type_', '').capitalize()
    subset = play_mae.filter(pl.col('play_type') == play_type)

    if len(subset) < 5:
        print(f"\n{play_type_clean}: Only {len(subset)} plays found (need 5)")
        continue

    print(f"\n{play_type_clean}:")
    print(f"  Total plays: {len(subset):,}")

    # Randomly sample 5 plays
    selected = subset.sample(n=5, seed=42)
    random_plays.append(selected)

    print(f"  Selected plays:")
    for i, row in enumerate(selected.sort('play_mae').iter_rows(named=True), 1):
        print(f"    {i}. Game {row['gameId']}, Play {row['playId']}: "
              f"MAE={row['play_mae']:.2f}y, Actual={row['yards_gained']}y")

if not random_plays:
    print("\n❌ No plays found to visualize!")
    exit(1)

random_plays_df = pl.concat(random_plays)
print(f"\n✓ Selected {len(random_plays_df)} plays total")

print("\nStep 7: Loading tracking data for visualization...")
# Load the prepped tracking data from gamestate_norm cache
tracking_path = PREPPED_DATA_PATH / "test_features.parquet"
if not tracking_path.exists():
    print(f"❌ Tracking data not found: {tracking_path}")
    exit(1)

tracking_df = pl.read_parquet(tracking_path)
print(f"✓ Loaded tracking data: {len(tracking_df):,} frames")

# Add yards_gained column (needed by visualize_play_v3)
# Note: 'expected_yards' already exists in results (weighted average from softmax)
joined = joined.with_columns(pl.col('true_yards').alias('yards_gained'))

print("\nStep 8: Creating visualizations (HTML animations + PNG graphs)...")
created_html = []
created_png = []

for row in random_plays_df.iter_rows(named=True):
    game_id = row['gameId']
    play_id = row['playId']
    play_type = row['play_type'].replace('play_type_', '')
    mae = row['play_mae']

    base_filename = f"{play_type}_game{game_id}_play{play_id}_mae{mae:.2f}"
    html_path = OUTPUT_DIR / f"{base_filename}.html"
    png_path = OUTPUT_DIR / f"{base_filename}.png"

    print(f"\n  Creating visualizations for {base_filename}...")

    # Create animation HTML
    fig = animate_play_with_predictions(tracking_df, joined, game_id, play_id)
    if fig is not None:
        fig.write_html(html_path)
        created_html.append(html_path)
        print(f"    ✓ Saved HTML: {html_path.name}")

    # Create PNG graph
    result_png = visualize_play_v3(joined, game_id, play_id, png_path, PREPPED_DATA_PATH)
    if result_png:
        created_png.append(result_png)
        print(f"    ✓ Saved PNG: {png_path.name}")

print("\n" + "="*80)
print("✅ VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nAll visualizations saved to: {OUTPUT_DIR.absolute()}")
print(f"Total HTML animations created: {len(created_html)}")
print(f"Total PNG graphs created: {len(created_png)}")
