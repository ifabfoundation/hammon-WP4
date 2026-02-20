#!/bin/bash
# Script per unire tutti i risultati LiDAR in un unico file
# Esegui dopo che tutti i job paralleli sono completati

set -euo pipefail

echo "=========================================="
echo "HAMMON - Merge LiDAR Heights Results"
echo "Start time: $(date)"
echo "=========================================="

cd /home/projects/hammon
export PYTHONPATH="/home/projects/hammon/src:${PYTHONPATH:-}"
PYTHON_BIN="/home/projects/hammon/hammon-env/bin/python"

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-YOUR_ACCESS_KEY_HERE}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-YOUR_SECRET_KEY_HERE}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="outputs/lidar/building_heights_complete_${TIMESTAMP}.geojson"

${PYTHON_BIN} << 'PYEOF'
import os
import geopandas as gpd
import pandas as pd
from src.common.s3_client import S3Client
from src.common.config_loader import ConfigLoader

# Lista delle zone
ZONES = [
    "centro1", "centro2", "est3", "est9", "nordovest1", "azno1", "azno2",
    "est7", "nordovest3", "azso2", "nordovest6", "azso3", "nordovest4", "est5",
    "azso4", "azso1", "azsc1", "nordovest5", "est6", "nordovest2", "est1",
    "est2", "est4", "sudest1", "sudest2", "est8", "sudcentrale", "azsc2"
]

config = ConfigLoader()
s3_cfg = config.get_s3_config()
s3 = S3Client(
    endpoint_url=s3_cfg['endpoint_url'],
    access_key=os.getenv('AWS_ACCESS_KEY_ID'),
    secret_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

print("Downloading and merging zone files...")
all_gdfs = []
for zone in ZONES:
    key = f"lidar_heights/building_heights_{zone}.geojson"
    try:
        gdf = s3.read_geodataframe("ifab.data", key)
        print(f"  {zone}: {len(gdf)} buildings")
        all_gdfs.append(gdf)
    except Exception as e:
        print(f"  {zone}: SKIP - {e}")

if all_gdfs:
    merged = pd.concat(all_gdfs, ignore_index=True)
    merged_gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs=all_gdfs[0].crs)
    
    # Save locally
    import sys
    timestamp = os.popen("date +%Y%m%d_%H%M%S").read().strip()
    output_path = f"outputs/lidar/building_heights_complete_{timestamp}.geojson"
    merged_gdf.to_file(output_path, driver="GeoJSON")
    print(f"\nSaved locally: {output_path}")
    print(f"Total buildings: {len(merged_gdf)}")
    
    # Upload to S3
    s3_key = "lidar_heights/building_heights_complete.geojson"
    s3.upload_file(output_path, "ifab.data", s3_key)
    print(f"Uploaded to: s3://ifab.data/{s3_key}")
    
    # Print summary stats
    print("\n=== SUMMARY ===")
    print(f"Total buildings with height: {merged_gdf['height_m'].notna().sum()}")
    print(f"Height range: {merged_gdf['height_m'].min():.1f}m - {merged_gdf['height_m'].max():.1f}m")
    print(f"Mean height: {merged_gdf['height_m'].mean():.1f}m")
else:
    print("ERROR: No files found to merge!")
    sys.exit(1)
PYEOF

echo "=========================================="
echo "Merge completed at $(date)"
echo "=========================================="
