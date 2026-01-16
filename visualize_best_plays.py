"""
Find and visualize 5 plays of each type from top 10% MAE from the 23-entity-football model.
For pass and rush plays, ensure at least 2 are above 10 yards.
Saves HTML visualizations to ./visualizations/ folder.
"""

import re
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
OUTPUT_DIR = Path('./visualizations')
OUTPUT_DIR.mkdir(exist_ok=True)

# Paths (adjust for your environment)
MODELS_DIR = Path('/content/drive/MyDrive/SportsTrackingTransformer/models_norm_football')
NGS_DATA_DIR = Path('/content/drive/MyDrive/NGS/NFL/REG/')
PREPPED_DATA_PATH = Path('/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_gamestate_norm')

PLAY_TYPES = ['play_type_pass', 'play_type_rush', 'play_type_sack']
MIN_YARDS = -10

print("="*80)
print("🏈 VISUALIZING BEST PREDICTIONS FROM 23-ENTITY-FOOTBALL MODEL")
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

def create_combined_visualization(
    animation_fig: go.Figure,
    test_results: pl.DataFrame,
    gameId: int,
    playId: int,
    output_path: Path,
):
    """
    Create an HTML file with the play animation on the left and prediction graphs on the right.
    """
    if animation_fig is None:
        print(f"  ⚠️  No animation figure for game {gameId}, play {playId}")
        return None

    # Get results for this play
    results_df = test_results.filter(
        (pl.col("gameId") == gameId) & (pl.col("playId") == playId)
    ).sort("frameId").to_pandas()

    if len(results_df) == 0:
        print(f"  ⚠️  No results data found for gameId={gameId}, playId={playId}")
        return None

    actual_yards = results_df["yards_gained"].iloc[0]
    results_df["absolute_error"] = (results_df["expected_yards"] - actual_yards).abs()

    # Extract title from animation figure
    title_text = animation_fig.layout.title.text if animation_fig.layout.title else ""

    # Create prediction graphs
    pred_fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Expected Yards Over Time", "Prediction Error Over Time"),
        vertical_spacing=0.12
    )

    # Top graph: Expected yards
    pred_fig.add_trace(
        go.Scatter(
            x=results_df["frameId"], y=results_df["expected_yards"],
            mode="lines+markers", name="Predicted",
            line=dict(color="#1E90FF", width=2), marker=dict(size=6)
        ),
        row=1, col=1
    )
    pred_fig.add_trace(
        go.Scatter(
            x=[results_df["frameId"].min(), results_df["frameId"].max()],
            y=[actual_yards, actual_yards],
            mode="lines", name=f"Actual: {actual_yards:.1f}",
            line=dict(color="red", width=2, dash="dash")
        ),
        row=1, col=1
    )

    # Bottom graph: Absolute error
    pred_fig.add_trace(
        go.Scatter(
            x=results_df["frameId"], y=results_df["absolute_error"],
            mode="lines+markers", name="Absolute Error",
            line=dict(color="#FF6347", width=2), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(255, 99, 71, 0.2)"
        ),
        row=2, col=1
    )
    pred_fig.add_trace(
        go.Scatter(
            x=[results_df["frameId"].min(), results_df["frameId"].max()],
            y=[0, 0],
            mode="lines", name="Perfect",
            line=dict(color="green", width=2, dash="dash")
        ),
        row=2, col=1
    )

    pred_fig.update_layout(
        height=800,
        paper_bgcolor="#333333",
        plot_bgcolor="#363636",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor="left"),
        margin=dict(l=60, r=100, t=60, b=40),
    )

    pred_fig.update_xaxes(title_text="Frame ID", gridcolor="rgba(255,255,255,0.1)")
    pred_fig.update_yaxes(title_text="Yards", gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
    pred_fig.update_yaxes(title_text="Error (yards)", gridcolor="rgba(255,255,255,0.1)", row=2, col=1)

    # Create combined HTML
    animation_html = animation_fig.to_html(full_html=False, include_plotlyjs=False)
    pred_html = pred_fig.to_html(full_html=False, include_plotlyjs=False)

    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Play Visualization - Game {gameId}, Play {playId}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                background-color: #222;
                color: white;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                display: flex;
                flex-direction: row;
                gap: 40px;
                width: 100%;
                max-width: 1900px;
                margin: 0 auto;
            }}
            .animation-panel {{
                flex: 0 0 55%;
                min-width: 600px;
            }}
            .graphs-panel {{
                flex: 0 0 40%;
                min-width: 500px;
                padding-left: 20px;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .legend-box {{
                background-color: #444;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                font-size: 12px;
            }}
            .legend-item {{
                display: inline-block;
                margin-right: 15px;
            }}
        </style>
    </head>
    <body>
        <h1>{title_text}</h1>

        <div class="container">
            <div class="animation-panel">
                <div class="legend-box">
                    <span class="legend-item"><span style="color: #89CFF0;">━━</span> LOS</span>
                    <span class="legend-item"><span style="color: #FFFF00;">━━</span> 1st Down</span>
                    <span class="legend-item"><span style="color: #00FF00;">━━</span> Actual</span>
                    <span class="legend-item"><span style="color: #FFD700;">┄┄</span> Predicted</span>
                </div>
                {animation_html}
            </div>

            <div class="graphs-panel">
                {pred_html}
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(combined_html)

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
    # For mirrored plays, join on gameId and playId only (no frameId match needed)
    # But add 'event' column as null to match schema
    play_meta = ngs_metadata.select(['gameId', 'playId', 'play_type']).unique()
    joined_mirror = test_mirror.join(play_meta, on=['gameId', 'playId'], how='left')
    joined_mirror = joined_mirror.with_columns(pl.lit(None).alias('event'))
    joined = pl.concat([joined_orig, joined_mirror], how="vertical_relaxed")
else:
    joined = joined_orig

print(f"✓ Joined: {len(joined):,} predictions with metadata")

print("\nStep 5: Computing MAE per play...")
# Compute MAE for each play
joined = joined.with_columns([
    (pl.col('yards_gained_class') + MIN_YARDS).alias('true_yards'),
    (pl.col('predicted_class') + MIN_YARDS).alias('pred_yards')
])

joined = joined.with_columns(
    (pl.col('true_yards') - pl.col('pred_yards')).abs().alias('abs_error')
)

# Get play-level MAE (average across all frames in the play)
play_mae = joined.group_by(['gameId', 'playId', 'play_type']).agg([
    pl.col('abs_error').mean().alias('play_mae'),
    pl.col('true_yards').first().alias('yards_gained'),
    pl.col('mirrored').first().alias('mirrored')
])

print(f"✓ Computed MAE for {len(play_mae):,} plays")

print("\nStep 6: Selecting plays from top 10% (with random sampling)...")
best_plays = []

for play_type in PLAY_TYPES:
    play_type_clean = play_type.replace('play_type_', '').capitalize()
    subset = play_mae.filter(pl.col('play_type') == play_type).sort('play_mae')

    if len(subset) < 5:
        print(f"\n{play_type_clean}: Only {len(subset)} plays found (need 5)")
        continue

    # Get top 10% by MAE
    top_10_pct_count = max(5, int(len(subset) * 0.10))
    top_10_pct = subset.head(top_10_pct_count)

    print(f"\n{play_type_clean}:")
    print(f"  Total plays: {len(subset):,}")
    print(f"  Top 10% size: {len(top_10_pct):,}")

    # For pass and rush, ensure at least 2 plays are above 10 yards
    if play_type in ['play_type_pass', 'play_type_rush']:
        above_10y = top_10_pct.filter(pl.col('yards_gained') > 10)
        below_or_eq_10y = top_10_pct.filter(pl.col('yards_gained') <= 10)

        print(f"  Plays >10y in top 10%: {len(above_10y)}")
        print(f"  Plays ≤10y in top 10%: {len(below_or_eq_10y)}")

        if len(above_10y) >= 2:
            # Sample 2 from above 10 yards, 3 from the rest
            sampled_above = above_10y.sample(n=2, seed=42)
            remaining_needed = 3

            if len(below_or_eq_10y) >= remaining_needed:
                sampled_below = below_or_eq_10y.sample(n=remaining_needed, seed=42)
            else:
                # Not enough below 10y, sample more from above 10y
                sampled_below = below_or_eq_10y
                additional_needed = remaining_needed - len(sampled_below)
                additional_above = above_10y.filter(
                    ~pl.col('playId').is_in(sampled_above['playId'])
                ).sample(n=min(additional_needed, len(above_10y) - 2), seed=42)
                sampled_below = pl.concat([sampled_below, additional_above])

            selected = pl.concat([sampled_above, sampled_below])
        else:
            # Not enough plays above 10 yards, just sample randomly
            print(f"  ⚠️  Only {len(above_10y)} plays >10y found, sampling randomly")
            selected = top_10_pct.sample(n=5, seed=42)
    else:
        # For sacks, just sample randomly from top 10%
        selected = top_10_pct.sample(n=5, seed=42)

    best_plays.append(selected)

    print(f"  Selected plays:")
    for i, row in enumerate(selected.sort('play_mae').iter_rows(named=True), 1):
        print(f"    {i}. Game {row['gameId']}, Play {row['playId']}: "
              f"MAE={row['play_mae']:.2f}y, Actual={row['yards_gained']}y")

if not best_plays:
    print("\n❌ No plays found to visualize!")
    exit(1)

best_plays_df = pl.concat(best_plays)
print(f"\n✓ Selected {len(best_plays_df)} plays total")

print("\nStep 7: Loading tracking data for visualization...")
# Load the prepped tracking data from gamestate_norm cache
tracking_path = PREPPED_DATA_PATH / "test_features.parquet"
if not tracking_path.exists():
    print(f"❌ Tracking data not found: {tracking_path}")
    exit(1)

tracking_df = pl.read_parquet(tracking_path)
print(f"✓ Loaded tracking data: {len(tracking_df):,} frames")

# Note: gamestate_norm has extra columns (yardsToGo, down, quarter, half_seconds_remaining)
# The visualization function will use what it needs - no filtering required
# since the animate function accesses columns by name, not position

# Add expected_yards column to joined data for visualization
joined = joined.with_columns(pl.col('pred_yards').alias('expected_yards'))

print("\nStep 8: Creating HTML visualizations...")
created_files = []

for row in best_plays_df.iter_rows(named=True):
    game_id = row['gameId']
    play_id = row['playId']
    play_type = row['play_type'].replace('play_type_', '')
    mae = row['play_mae']

    output_filename = f"{play_type}_game{game_id}_play{play_id}_mae{mae:.2f}.html"
    output_path = OUTPUT_DIR / output_filename

    print(f"\n  Creating {output_filename}...")

    # Create animation
    fig = animate_play_with_predictions(tracking_df, joined, game_id, play_id)

    if fig is not None:
        # Create combined visualization
        result_path = create_combined_visualization(fig, joined, game_id, play_id, output_path)
        if result_path:
            created_files.append(result_path)
            print(f"    ✓ Saved: {output_filename}")

print("\n" + "="*80)
print("✅ VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nAll visualizations saved to: {OUTPUT_DIR.absolute()}")
print(f"Total HTML files created: {len(created_files)}")
