#!/bin/bash
#SBATCH --job-name=lidar_rem
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lidar_rem_%A_%a.out
#SBATCH --error=logs/lidar_rem_%A_%a.err
#SBATCH --array=1-4

set -euo pipefail

# Zone mancanti - 4 zone che hanno fatto timeout o failed
ZONES=(
    "centro1"       # 1  - 361 edifici mancanti (da run 41 in poi)
    "centro2"       # 2  - 176 edifici mancanti (da run 67 in poi)
    "est3"          # 3  - 25 edifici mancanti (da run 85 in poi)
    "azso2"         # 4  - 33 edifici mancanti (da run 90 in poi, skip 89 corrotto)
)

# Array index è 1-based
ZONE="${ZONES[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "=========================================="
echo "HAMMON - LiDAR Height Extraction (REMAINING)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID / 4"
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

OUTPUT_FILE="outputs/lidar/building_heights_${ZONE}_remaining.geojson"
S3_KEY="lidar_heights/building_heights_${ZONE}_remaining.geojson"

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
