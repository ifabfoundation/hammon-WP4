#!/bin/bash
#SBATCH --job-name=lidar_par
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/lidar_par_%A_%a.out
#SBATCH --error=logs/lidar_par_%A_%a.err
#SBATCH --array=1-8

set -euo pipefail

# Zone assignments per job
declare -A ZONE_MAP
ZONE_MAP[1]="centro1"
ZONE_MAP[2]="centro2"
ZONE_MAP[3]="est3,est9"
ZONE_MAP[4]="nordovest1,azno1,azno2"
ZONE_MAP[5]="est7,nordovest3,azso2"
ZONE_MAP[6]="nordovest6,azso3,nordovest4,est5"
ZONE_MAP[7]="azso4,azso1,azsc1,nordovest5,est6"
ZONE_MAP[8]="nordovest2,est1,est2,est4,sudest1,sudest2,est8,sudcentrale,azsc2"

ZONES="${ZONE_MAP[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "HAMMON - LiDAR Height Extraction (PARALLEL)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Zones: $ZONES"
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
OUTPUT_FILE="outputs/lidar/building_heights_job${SLURM_ARRAY_TASK_ID}_${TIMESTAMP}.geojson"
S3_KEY="lidar_heights/building_heights_job${SLURM_ARRAY_TASK_ID}.geojson"

# Process each zone in this job's assignment
IFS=',' read -ra ZONE_ARRAY <<< "$ZONES"
PROCESSED_PARTS=()

for ZONE in "${ZONE_ARRAY[@]}"; do
    echo "Processing zone: $ZONE"
    ZONE_OUTPUT="outputs/lidar/building_heights_${ZONE}_${TIMESTAMP}.geojson"
    
    ${PYTHON_BIN} -m src.lidar.batch_runner \
      --gdf-bucket "ifab.data" \
      --gdf-key "geodataframe/facades_matched_complete.geojson" \
      --zone "${ZONE}" \
      --radii "3,5,8" \
      --output "${ZONE_OUTPUT}" \
      --output-format "geojson" \
      --temp-dir "${TMPDIR_LIDAR}" \
      --upload-s3 \
      --s3-output-bucket "ifab.data" \
      --s3-output-key "lidar_heights/building_heights_${ZONE}.geojson"
    
    PROCESSED_PARTS+=("${ZONE_OUTPUT}")
    echo "Completed zone: $ZONE"
done

# Cleanup temp directory
rm -rf "${TMPDIR_LIDAR}"

echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID completed"
echo "Processed zones: $ZONES"
echo "Done at $(date)"
echo "=========================================="
