#!/bin/bash
# Sync target files from add-game-state-features branch cache to 23-entity branch

# Source: prepped data from add-game-state-features branch (targets are the same)
SOURCE_DIR="/content/drive/MyDrive/ExtraDataSportsTrackingTransformer_cache_gamestate"

# Destination: prepped data for 23-entity branch
DEST_DIR="data/split_prepped_data_extra"

echo "============================================================"
echo "Syncing target files from Google Drive"
echo "============================================================"
echo ""
echo "Source: $SOURCE_DIR"
echo "Dest:   $DEST_DIR"
echo ""

# Create destination directory
mkdir -p "$DEST_DIR"

# Copy target files (these are the same across branches)
for split in train val test; do
    source_file="$SOURCE_DIR/${split}_targets.parquet"
    dest_file="$DEST_DIR/${split}_targets.parquet"

    if [ -f "$dest_file" ]; then
        size=$(du -h "$dest_file" | cut -f1)
        echo "✓ Already exists: $dest_file ($size)"
    elif [ -f "$source_file" ]; then
        echo "📁 Copying: $source_file"
        cp "$source_file" "$dest_file"
        size=$(du -h "$dest_file" | cut -f1)
        echo "   → Saved to: $dest_file ($size)"
    else
        echo "❌ Missing: $source_file"
    fi
    echo ""
done

echo "============================================================"
echo "Done!"
echo "============================================================"
