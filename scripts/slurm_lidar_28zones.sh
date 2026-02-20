#!/bin/bash
#SBATCH --job-name=lidar_z
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/lidar_z_%A_%a.out
#SBATCH --error=logs/lidar_z_%A_%a.err
#SBATCH --array=1-28%5

set -euo pipefail

# Una zona per job - 28 zone totali
ZONES=(
    "centro1"       # 1  - 1103 edifici, 161 runs
    "centro2"       # 2  - 845 edifici, 130 runs
    "est3"          # 3  - 635 edifici, 154 runs
    "nordovest1"    # 4  - 456 edifici, 117 runs
    "azno1"         # 5  - 400 edifici, 101 runs
    "est9"          # 6  - 368 edifici, 101 runs
    "est7"          # 7  - 354 edifici, 110 runs
    "nordovest3"    # 8  - 323 edifici, 76 runs
    "azso2"         # 9  - 255 edifici, 82 runs
    "nordovest6"    # 10 - 234 edifici, 72 runs
    "azno2"         # 11 - 216 edifici, 60 runs
    "azso3"         # 12 - 215 edifici, 82 runs
    "nordovest4"    # 13 - 203 edifici, 69 runs
    "est5"          # 14 - 188 edifici, 46 runs
    "azso4"         # 15 - 187 edifici, 68 runs
    "azso1"         # 16 - 179 edifici, 25 runs
    "azsc1"         # 17 - 168 edifici, 57 runs
    "nordovest5"    # 18 - 167 edifici, 41 runs
    "est6"          # 19 - 163 edifici, 47 runs
    "nordovest2"    # 20 - 142 edifici, 37 runs
    "est1"          # 21 - 130 edifici, 51 runs
    "est4"          # 22 - 126 edifici, 34 runs
    "est2"          # 23 - 121 edifici, 45 runs
    "sudest1"       # 24 - 118 edifici, 45 runs
    "sudest2"       # 25 - 107 edifici, 41 runs
    "est8"          # 26 - 90 edifici, 35 runs
    "sudcentrale"   # 27 - 43 edifici, 11 runs
    "azsc2"         # 28 - 22 edifici, 13 runs
)

# Array index è 1-based
ZONE="${ZONES[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "=========================================="
echo "HAMMON - LiDAR Height Extraction"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID / 28"
echo "Zone: $ZONE"
echo "Start time: $(date)"
echo "=========================================="

cd /home/projects/hammon
mkdir -p logs outputs/lidar
TMPDIR_LIDAR="/home/projects/hammon/data/tmp_lidar_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR_LIDAR}"

export PYTHONPATH="/home/projects/hammon/src:${PYTHONPATH:-}"
PYTHON_BIN="/home/projects/hammon/hammon-env/bin/python"

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-YOUR_ACCESS_KEY_HERE}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-YOUR_SECRET_KEY_HERE}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="outputs/lidar/building_heights_${ZONE}.geojson"
S3_KEY="lidar_heights/building_heights_${ZONE}.geojson"

${PYTHON_BIN} -m src.lidar.batch_runner \
  --gdf-bucket "ifab.data" \
  --gdf-key "geodataframe/facades_matched_complete.geojson" \
  --zone "${ZONE}" \
  --radii "3,5,8" \
  --output "${OUTPUT_FILE}" \
  --output-format "geojson" \
  --temp-dir "${TMPDIR_LIDAR}" \
  --upload-s3 \
  --s3-output-bucket "ifab.data" \
  --s3-output-key "${S3_KEY}"

# Cleanup temp directory
rm -rf "${TMPDIR_LIDAR}"

echo "=========================================="
echo "Zone $ZONE completed!"
echo "Output: s3://ifab.data/${S3_KEY}"
echo "Done at $(date)"
echo "=========================================="
