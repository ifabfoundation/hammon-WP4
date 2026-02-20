#!/bin/bash
#SBATCH --job-name=lidar_heights
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/lidar_heights_%j.out
#SBATCH --error=logs/lidar_heights_%j.err

set -euo pipefail

echo "=========================================="
echo "HAMMON - LiDAR Height Extraction (FULL)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Zone filter: ${ZONE:-all}"
echo "Run filter: ${RUNS:-all}"
echo "=========================================="

cd /home/projects/hammon
mkdir -p logs outputs/lidar
TMPDIR_LIDAR=${SLURM_TMPDIR:-/home/projects/hammon/data/tmp_lidar_${SLURM_JOB_ID:-run}}
mkdir -p "${TMPDIR_LIDAR}"

# Ensure Python can find the src/ modules
export PYTHONPATH="/home/projects/hammon/src:${PYTHONPATH:-}"

PYTHON_BIN="/home/projects/hammon/hammon-env/bin/python"

srun_env_creds() {
  export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-YOUR_ACCESS_KEY_HERE}"
  export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-YOUR_SECRET_KEY_HERE}"
}

srun_env_creds

EXTRA_ARGS=()
if [[ -n "${ZONE:-}" ]]; then
  EXTRA_ARGS+=(--zone "${ZONE}")
fi
if [[ -n "${RUNS:-}" ]]; then
  EXTRA_ARGS+=(--runs "${RUNS}")
fi

# Generate output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="outputs/lidar/building_heights_${TIMESTAMP}.geojson"
S3_KEY="lidar_heights/building_heights_${TIMESTAMP}.geojson"

${PYTHON_BIN} -m src.lidar.batch_runner \
  --gdf-bucket "ifab.data" \
  --gdf-key "geodataframe/facades_matched_complete.geojson" \
  --radii "3,5,8" \
  --output "${OUTPUT_FILE}" \
  --output-format "geojson" \
  --temp-dir "${TMPDIR_LIDAR}" \
  --upload-s3 \
  --s3-output-bucket "ifab.data" \
  --s3-output-key "${S3_KEY}" \
  "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "Output file: ${OUTPUT_FILE}"
echo "Uploaded to: s3://ifab.data/${S3_KEY}"
echo "Done at $(date)"
echo "=========================================="
